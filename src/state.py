"""
Agent state model.

The agent state at time t is the tuple:

    S_t = (P_t, M_t, B_t, G_t)

where:
    P_t  -- Control-plane state (permissions, integrations, policies)
    M_t  -- Memory / persistent storage
    B_t  -- Behaviour parameters (system prompt extensions, reminders)
    G_t  -- IR graph capturing all information flow

The control plane P_t is further decomposed as:
    P_t = (Perm_t, Int_t, Policy_t)

    Perm_t   -- set of allowed tools and actions
    Int_t    -- configured chat channels, API connections, skill installations
    Policy_t -- mappings from principals to permitted scope
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Set

from .ir import IRGraph
from .principals import Principal


# ── Control-plane components ─────────────────────────────────────────────

@dataclass
class Permission:
    """A single permission entry."""
    tool_name: str
    allowed_actions: FrozenSet[str]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Permission):
            return NotImplemented
        return (self.tool_name == other.tool_name
                and self.allowed_actions == other.allowed_actions)

    def __hash__(self) -> int:
        return hash((self.tool_name, self.allowed_actions))


@dataclass
class Integration:
    """A configured external integration (chat channel, API, skill)."""
    name: str
    kind: str           # e.g. "chat_channel", "api_connection", "skill"
    config: Dict[str, str] = field(default_factory=dict)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Integration):
            return NotImplemented
        return (self.name == other.name
                and self.kind == other.kind
                and self.config == other.config)

    def __hash__(self) -> int:
        return hash((self.name, self.kind, tuple(sorted(self.config.items()))))


@dataclass
class Policy:
    """A scope policy mapping a principal to permitted directories/resources."""
    principal: Principal
    allowed_paths: FrozenSet[str]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Policy):
            return NotImplemented
        return (self.principal == other.principal
                and self.allowed_paths == other.allowed_paths)

    def __hash__(self) -> int:
        return hash((self.principal, self.allowed_paths))


# ── Control-plane aggregate ──────────────────────────────────────────────

@dataclass
class ControlPlane:
    """P_t = (Perm_t, Int_t, Policy_t)."""
    permissions: Set[Permission] = field(default_factory=set)
    integrations: Set[Integration] = field(default_factory=set)
    policies: Set[Policy] = field(default_factory=set)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ControlPlane):
            return NotImplemented
        return (self.permissions == other.permissions
                and self.integrations == other.integrations
                and self.policies == other.policies)

    def copy(self) -> ControlPlane:
        return copy.deepcopy(self)


# ── Full agent state ─────────────────────────────────────────────────────

@dataclass
class AgentState:
    """S_t = (P_t, M_t, B_t, G_t)."""
    control_plane: ControlPlane = field(default_factory=ControlPlane)
    memory: Dict[str, str] = field(default_factory=dict)        # M_t
    behaviour: Dict[str, str] = field(default_factory=dict)     # B_t
    ir_graph: IRGraph = field(default_factory=IRGraph)          # G_t
    step: int = 0                                                # t

    def copy(self) -> AgentState:
        return copy.deepcopy(self)
