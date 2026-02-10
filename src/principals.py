"""
Principal hierarchy and trust classification.

Principals model the origin of information flowing through the agent.
Trusted principals (SYS, USER) may authorise control-plane changes;
untrusted principals (WEB, SKILL, TOOL_OUTPUT) may not.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import FrozenSet


class Principal(Enum):
    """Origin tag attached to every piece of information in the agent."""

    SYS = auto()          # System / platform policies
    USER = auto()         # Direct, confirmed user input
    WEB = auto()          # Content scraped from the web or emails
    SKILL = auto()        # Content originating from an installed skill file
    TOOL_OUTPUT = auto()  # Return value of an external tool invocation

    @property
    def is_trusted(self) -> bool:
        return self in _TRUSTED_PRINCIPALS


_TRUSTED_PRINCIPALS: FrozenSet[Principal] = frozenset({
    Principal.SYS,
    Principal.USER,
})

UNTRUSTED_PRINCIPALS: FrozenSet[Principal] = frozenset({
    Principal.WEB,
    Principal.SKILL,
    Principal.TOOL_OUTPUT,
})


def is_trusted_principal(p: Principal) -> bool:
    """Return True iff *p* is allowed to justify control-plane mutations."""
    return p in _TRUSTED_PRINCIPALS
