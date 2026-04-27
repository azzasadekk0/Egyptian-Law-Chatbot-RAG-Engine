import time
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

MEMORY_MAX_TURNS         = 10
FOLLOWUP_TOKEN_THRESHOLD = 8

# Arabic particles/conjunctions that strongly indicate a follow-up question
_FOLLOWUP_PREFIXES: tuple[str, ...] = (
    "ولو",    # and if
    "وإذا",   # and if (formal)
    "وهل",    # and is/does
    "وكيف",   # and how
    "وماذا",  # and what
    "ومتى",   # and when
    "ولكن",   # but
    "وما ",   # and what (space prevents matching وماذا twice)
    "فماذا",  # so what
    "فهل",    # so is
    "فكيف",   # so how
    "ثم ",    # then (space required)
    "أما ",   # as for
    "وأيضاً", # and also
    "وكذلك",  # and likewise
)


@dataclass
class Turn:
    """A single question-answer exchange."""
    question:  str
    answer:    str
    law_type:  str
    timestamp: float = field(default_factory=time.time)

    def format_for_context(self, max_answer_chars: int = 400) -> str:
        answer_preview = (
            self.answer[:max_answer_chars] + "…"
            if len(self.answer) > max_answer_chars
            else self.answer
        )
        return f"المستخدم: {self.question}\nالمساعد: {answer_preview}"


class ConversationMemory:
    """
    In-process memory store for all active sessions.
    One instance is shared across all sessions in a single EgyptianLegalRAG object.
    """

    def __init__(self, max_turns: int = MEMORY_MAX_TURNS) -> None:
        self.max_turns = max_turns
        self._sessions: dict[str, deque[Turn]] = {}

    def add_turn(self, session_id: str, question: str, answer: str, law_type: str) -> None:
        """Append a completed turn to this session's history."""
        if session_id not in self._sessions:
            self._sessions[session_id] = deque(maxlen=self.max_turns)
        self._sessions[session_id].append(Turn(question=question, answer=answer, law_type=law_type))
        logger.debug(f"[Memory] Session '{session_id}' — stored turn #{len(self._sessions[session_id])}")

    def clear_session(self, session_id: str) -> None:
        """Discard all history for this session."""
        self._sessions.pop(session_id, None)

    def get_turns(self, session_id: str) -> list[Turn]:
        """Return all stored turns for this session (oldest first)."""
        return list(self._sessions.get(session_id, []))

    def has_history(self, session_id: str) -> bool:
        """True if this session has at least one prior turn."""
        return bool(self._sessions.get(session_id))

    def get_last_law_type(self, session_id: str) -> Optional[str]:
        """Return the most recent known domain, skipping 'unknown' turns."""
        for turn in reversed(self.get_turns(session_id)):
            if turn.law_type and turn.law_type != "unknown":
                return turn.law_type
        return None

    def format_context(self, session_id: str, max_turns: int = 5, max_answer_chars: int = 400) -> str:
        """Build a formatted conversation context string for LLM injection."""
        turns = self.get_turns(session_id)
        if not turns:
            return ""
        blocks = [t.format_for_context(max_answer_chars) for t in turns[-max_turns:]]
        return "\n\n".join(blocks)

    def is_followup(self, session_id: str, question: str) -> bool:
        """
        Heuristic follow-up detector — no LLM call, runs in < 1ms.
        Returns True when the session has history, the query is short (< 8 tokens),
        and it starts with a known Arabic conjunction or particle.
        """
        if not self.has_history(session_id):
            return False
        tokens = question.strip().split()
        if len(tokens) >= FOLLOWUP_TOKEN_THRESHOLD:
            return False
        q = question.strip()
        for prefix in _FOLLOWUP_PREFIXES:
            if q.startswith(prefix):
                logger.debug(f"[Memory] Follow-up detected: '{question[:50]}' (prefix='{prefix}')")
                return True
        return False

    @property
    def session_count(self) -> int:
        """Number of active sessions currently in memory."""
        return len(self._sessions)
