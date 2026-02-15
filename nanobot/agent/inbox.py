"""Inbox agent loop: deterministic message-to-Obsidian routing.

Extends AgentLoop to intercept messages from configured inbox chats
and write them directly to Obsidian without LLM calls. Messages that
don't match any keyword pattern fall through to the normal agent.
"""

import re
from datetime import datetime, timedelta
from pathlib import Path

from loguru import logger

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage, OutboundMessage


# ---------------------------------------------------------------------------
# Intent classification: keyword patterns (German)
# ---------------------------------------------------------------------------

KEYWORD_PATTERNS: dict[str, list[str]] = {
    "journal": [
        r"\bheute\b", r"\btag\b", r"\berlebt\b", r"\bgemacht\b",
        r"\bmorgens\b", r"\babends\b", r"\bgestern\b", r"\bwar\b",
    ],
    "ideen": [
        r"\bidee\b", r"\bkönnte\b", r"\bprojekt\b", r"\bwas wäre\b",
    ],
    "reflexion": [
        r"\bdenke\b", r"\bfrage mich\b", r"\bwarum\b",
        r"\bbedeutet\b", r"\bfühle\b",
    ],
}

# ---------------------------------------------------------------------------
# Output configuration per intent
# ---------------------------------------------------------------------------

OUTPUT_CONFIG: dict[str, dict] = {
    "journal": {
        "folder": "1 🗓️ Journals/{year}/{month:02d}",
        "filename": "{date}.md",
        "mode": "append",
        "template": "\n## {time} Uhr\n\n{text}\n",
        "emoji": "📝",
    },
    "ideen": {
        "folder": "2 💡 Ideen",
        "filename": "{timestamp}_{title}.md",
        "mode": "create",
        "template": "# 💡 {title}\n\nErstellt: {datetime}\n\n---\n\n{text}\n",
        "emoji": "💡",
    },
    "reflexion": {
        "folder": "3 🤔 Reflexionen",
        "filename": "{date}_{title}.md",
        "mode": "create",
        "template": "# 🤔 {title}\n\nErstellt: {datetime}\n\n---\n\n{text}\n",
        "emoji": "🤔",
    },
    "info": {
        "folder": "4 📝 Notizen",
        "filename": "{timestamp}_{title}.md",
        "mode": "create",
        "template": "# 📝 {title}\n\nErstellt: {datetime}\n\n---\n\n{text}\n",
        "emoji": "📝",
    },
}


# ---------------------------------------------------------------------------
# InboxAgentLoop
# ---------------------------------------------------------------------------

class InboxAgentLoop(AgentLoop):
    """AgentLoop subclass that adds deterministic inbox processing.

    Messages from configured inbox chat IDs are classified via keyword
    matching and written directly to Obsidian.  Messages that don't
    match any keyword pattern fall through to the normal LLM agent.
    Messages from non-inbox chats are always handled by the normal agent.
    """

    def __init__(self, *args, inbox_config: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = inbox_config or {}
        self.inbox_chat_ids: set[str] = set(cfg.get("inbox_chat_ids", []))
        self.obsidian_root: Path = Path(cfg.get("obsidian_root", "~/Documents/Obsidian")).expanduser()
        self.day_cutoff_hour: int = cfg.get("day_cutoff_hour", 4)

        logger.info(
            f"Inbox enabled: {len(self.inbox_chat_ids)} chat(s), "
            f"root={self.obsidian_root}, cutoff={self.day_cutoff_hour}h"
        )

    # -- override -----------------------------------------------------------

    async def _process_message(
        self, msg: InboundMessage, session_key: str | None = None
    ) -> OutboundMessage | None:
        """Route inbox messages to deterministic processing, rest to LLM."""
        if msg.chat_id in self.inbox_chat_ids:
            result = await self._process_inbox(msg)
            if result is not None:
                return result
            # No keyword match → fall through to normal agent
            logger.info(
                f"Inbox: no pattern match for '{msg.content[:50]}…', "
                "delegating to LLM"
            )

        return await super()._process_message(msg, session_key)

    # -- inbox pipeline -----------------------------------------------------

    async def _process_inbox(self, msg: InboundMessage) -> OutboundMessage | None:
        """Classify and write to Obsidian. Returns None if no keyword match."""
        intent = self._classify(msg.content)
        if intent is None:
            return None

        path = self._write_to_obsidian(msg.content, intent)
        config = OUTPUT_CONFIG[intent]
        emoji = config["emoji"]
        reply = f"{emoji} {intent.capitalize()} gespeichert → {path.name}"

        logger.info(f"Inbox: {intent} → {path}")
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=reply,
        )

    # -- helpers ------------------------------------------------------------

    def _classify(self, text: str) -> str | None:
        """Keyword-based classification. Returns None if no match → LLM fallback."""
        text_lower = text.lower()
        for intent, patterns in KEYWORD_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return intent
        return None

    def _write_to_obsidian(self, text: str, intent: str) -> Path:
        """Write content to the correct Obsidian folder. Returns the file path."""
        now = self._effective_date()
        config = OUTPUT_CONFIG[intent]
        title = self._safe_title(text)

        # Build folder path
        folder_pattern = config["folder"]
        folder = self.obsidian_root / folder_pattern.format(
            year=now.year, month=now.month
        )
        folder.mkdir(parents=True, exist_ok=True)

        # Build filename
        filename = config["filename"].format(
            date=now.strftime("%Y-%m-%d"),
            timestamp=now.strftime("%Y%m%d_%H%M"),
            title=title,
        )
        path = folder / filename

        # Build content from template
        content = config["template"].format(
            text=text,
            title=title,
            time=now.strftime("%H:%M"),
            date=now.strftime("%Y-%m-%d"),
            datetime=now.isoformat(),
            timestamp=now.strftime("%Y%m%d_%H%M"),
        )

        # Write
        if config["mode"] == "append":
            with open(path, "a", encoding="utf-8") as f:
                f.write(content)
        else:
            path.write_text(content, encoding="utf-8")

        return path

    def _effective_date(self) -> datetime:
        """Before cutoff hour → use previous day (night entries belong to the day before)."""
        now = datetime.now()
        if now.hour < self.day_cutoff_hour:
            now -= timedelta(days=1)
        return now

    @staticmethod
    def _safe_title(text: str) -> str:
        """Extract first line as title, make filesystem-safe."""
        first_line = text.strip().split("\n")[0][:50]
        safe = re.sub(r'[^\w\s-]', '', first_line).strip()
        return safe[:40] or "notiz"
