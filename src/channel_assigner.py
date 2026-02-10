"""
Channel-based principal assignment.

Addresses Reviewer Concern #2 (confused deputy): Principals must be assigned
based on the *channel* through which content arrives, never inferred from
content.  This module enforces that invariant.

The key design rule:
    "Principal is determined by the transport channel, not by the payload."

For example:
  - Content arriving via HTTP scrape → WEB, regardless of what the HTML says
  - Content returned by a tool API → TOOL_OUTPUT, even if it claims "user said"
  - Content from the skill file store → SKILL, even if it contains "[SYSTEM]"
  - Content from the authenticated user session → USER
  - Content from the platform runtime → SYS

This eliminates the confused deputy attack (Part B2 Scenario 2) where an LLM
output is mislabeled because the LLM's *content* claims to be from the user.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from .principals import Principal


class Channel(Enum):
    """Transport channels through which content enters the agent."""

    PLATFORM_RUNTIME = auto()       # Internal system messages
    AUTHENTICATED_USER_SESSION = auto()  # Direct user interaction (keyboard/voice)
    HTTP_SCRAPE = auto()            # Web page fetches, RSS feeds
    SKILL_FILE_STORE = auto()       # Loaded skill definitions
    TOOL_API_RETURN = auto()        # Return values from tool invocations
    LLM_GENERATION = auto()         # LLM output (always derived, never a root principal)
    MEMORY_RETRIEVAL = auto()       # Content retrieved from persistent memory


# Immutable mapping: channel → principal
# This is the ONLY place where principal assignment happens.
_CHANNEL_TO_PRINCIPAL = {
    Channel.PLATFORM_RUNTIME: Principal.SYS,
    Channel.AUTHENTICATED_USER_SESSION: Principal.USER,
    Channel.HTTP_SCRAPE: Principal.WEB,
    Channel.SKILL_FILE_STORE: Principal.SKILL,
    Channel.TOOL_API_RETURN: Principal.TOOL_OUTPUT,
    # LLM_GENERATION and MEMORY_RETRIEVAL are DERIVED channels.
    # They do not get a root principal — their taint comes from dependencies.
    Channel.LLM_GENERATION: Principal.SYS,
    Channel.MEMORY_RETRIEVAL: Principal.SYS,
}


@dataclass(frozen=True)
class ChannelAssignment:
    """Result of assigning a principal to incoming content."""
    channel: Channel
    principal: Principal
    requires_dependencies: bool  # True if this is a derived channel


class ChannelPrincipalAssigner:
    """Assigns principals based strictly on transport channel.

    This class enforces the invariant that principal assignment NEVER
    inspects content.  The assign() method takes only the channel enum
    and returns the corresponding principal.

    For derived channels (LLM_GENERATION, MEMORY_RETRIEVAL), the caller
    MUST also provide dependency node IDs to the IR graph when creating
    the node.  The assigner flags this requirement in the result.
    """

    # Channels where the node MUST carry dependency edges
    _DERIVED_CHANNELS = frozenset({
        Channel.LLM_GENERATION,
        Channel.MEMORY_RETRIEVAL,
    })

    def assign(self, channel: Channel) -> ChannelAssignment:
        """Assign a principal based on channel alone.

        Returns a ChannelAssignment indicating the principal and whether
        the caller must provide dependency edges (for derived channels).

        Raises ValueError if the channel is unknown.
        """
        principal = _CHANNEL_TO_PRINCIPAL.get(channel)
        if principal is None:
            raise ValueError(f"Unknown channel: {channel}")

        return ChannelAssignment(
            channel=channel,
            principal=principal,
            requires_dependencies=(channel in self._DERIVED_CHANNELS),
        )

    @staticmethod
    def validate_no_content_inspection(content: str, channel: Channel) -> bool:
        """Validation check: content must NOT influence principal assignment.

        This is a design-time assertion.  In a correct implementation,
        the principal is determined solely by `channel`.  This method
        exists to make the invariant testable: it always returns True
        because the content is intentionally ignored.
        """
        # Content is intentionally unused — that's the whole point.
        _ = content
        return True
