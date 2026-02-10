"""
Agent State Model for Control-Plane Integrity Evaluation

This module implements the formal state model from the theorem:
S_t = (P_t, M_t, B_t, G_t)

Where:
- P_t: Control plane (permissions, integrations, policies)
- M_t: Memory state
- B_t: Belief state
- G_t: Goal state
"""

from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional, Any
from enum import Enum
from datetime import datetime
import json
import hashlib


class Principal(Enum):
    """Principal types for provenance tracking."""
    SYS = "system"
    USER = "user"
    WEB = "web"
    SKILL = "skill"
    TOOL = "tool"


class TaintStatus(Enum):
    """Taint status for IR nodes."""
    UNTAINTED = 0  # tau = 0
    TAINTED = 1    # tau = 1


@dataclass
class Permission:
    """Represents a permission in the control plane."""
    tool_name: str
    action: str
    scope: Optional[Dict[str, Any]] = None
    granted_at: datetime = field(default_factory=datetime.now)
    granted_by: Principal = Principal.USER

    def __hash__(self):
        return hash((self.tool_name, self.action, json.dumps(self.scope, sort_keys=True)))

    def __eq__(self, other):
        if not isinstance(other, Permission):
            return NotImplemented
        return (self.tool_name == other.tool_name
                and self.action == other.action
                and self.scope == other.scope)


@dataclass
class Integration:
    """Represents an integration (chat channel, API connection, skill)."""
    integration_type: str  # e.g., "telegram", "slack", "discord"
    name: str
    config: Dict[str, Any] = field(default_factory=dict)
    installed_at: datetime = field(default_factory=datetime.now)
    installed_by: Principal = Principal.USER

    def __hash__(self):
        return hash((self.integration_type, self.name, json.dumps(self.config, sort_keys=True)))

    def __eq__(self, other):
        if not isinstance(other, Integration):
            return NotImplemented
        return (self.integration_type == other.integration_type
                and self.name == other.name
                and self.config == other.config)


@dataclass
class Policy:
    """Represents a policy mapping principals to permitted scope."""
    principal: Principal
    resource: str
    allowed_actions: Set[str] = field(default_factory=set)
    constraints: Optional[Dict[str, Any]] = None

    def __hash__(self):
        return hash((self.principal, self.resource, frozenset(self.allowed_actions)))

    def __eq__(self, other):
        if not isinstance(other, Policy):
            return NotImplemented
        return (self.principal == other.principal
                and self.resource == other.resource
                and self.allowed_actions == other.allowed_actions)


@dataclass
class ControlPlane:
    """Control plane state P_t.

    Contains:
    - Permissions: Set of allowed tools and actions
    - Integrations: Configured chat channels, APIs, skills
    - Policies: Mappings from principals to permitted scope
    """
    permissions: Set[Permission] = field(default_factory=set)
    integrations: Set[Integration] = field(default_factory=set)
    policies: Set[Policy] = field(default_factory=set)

    def add_permission(self, permission: Permission) -> bool:
        self.permissions.add(permission)
        return True

    def remove_permission(self, permission: Permission) -> bool:
        if permission in self.permissions:
            self.permissions.remove(permission)
            return True
        return False

    def add_integration(self, integration: Integration) -> bool:
        self.integrations.add(integration)
        return True

    def remove_integration(self, integration: Integration) -> bool:
        if integration in self.integrations:
            self.integrations.remove(integration)
            return True
        return False

    def add_policy(self, policy: Policy) -> bool:
        self.policies.add(policy)
        return True

    def has_permission(self, tool_name: str, action: str) -> bool:
        return any(p.tool_name == tool_name and p.action == action for p in self.permissions)

    def has_integration(self, integration_type: str, name: str) -> bool:
        return any(i.integration_type == integration_type and i.name == name
                   for i in self.integrations)

    def get_state_hash(self) -> str:
        state_repr = {
            "permissions": sorted([f"{p.tool_name}:{p.action}" for p in self.permissions]),
            "integrations": sorted([f"{i.integration_type}:{i.name}" for i in self.integrations]),
            "policies": sorted([f"{p.principal.value}:{p.resource}" for p in self.policies]),
        }
        return hashlib.sha256(json.dumps(state_repr, sort_keys=True).encode()).hexdigest()

    def __eq__(self, other):
        if not isinstance(other, ControlPlane):
            return False
        return (self.permissions == other.permissions
                and self.integrations == other.integrations
                and self.policies == other.policies)


@dataclass
class IRNode:
    """Intermediate Representation (IR) Node.

    Tracks provenance and taint status for reasoning steps.
    """
    content: str
    principal: Principal
    taint: TaintStatus
    timestamp: datetime = field(default_factory=datetime.now)
    dependencies: List[str] = field(default_factory=list)
    node_id: str = field(
        default_factory=lambda: hashlib.sha256(str(datetime.now().timestamp()).encode()).hexdigest()[:16]
    )

    def is_untainted(self) -> bool:
        return self.taint == TaintStatus.UNTAINTED

    def is_trusted(self) -> bool:
        return self.principal in (Principal.SYS, Principal.USER)


@dataclass
class JustificationCertificate:
    """Certificate for control-plane update proposals.

    Lists IR nodes used to derive the change and their principal authority.
    """
    proposal_id: str
    proposed_change: Dict[str, Any]
    dependency_nodes: List[IRNode]
    effective_principal: Principal
    timestamp: datetime = field(default_factory=datetime.now)

    def verify(self) -> bool:
        """Verify certificate according to theorem requirements:
        1. All nodes in dependency set must be untainted (tau = 0)
        2. Effective principal must be SYS or confirmed USER
        """
        all_untainted = all(node.is_untainted() for node in self.dependency_nodes)
        trusted_principal = self.effective_principal in (Principal.SYS, Principal.USER)
        return all_untainted and trusted_principal


@dataclass
class MemoryState:
    """Memory state M_t."""
    short_term: List[str] = field(default_factory=list)
    long_term: Dict[str, Any] = field(default_factory=dict)
    persistent_storage: Dict[str, str] = field(default_factory=dict)


@dataclass
class BeliefState:
    """Belief state B_t."""
    beliefs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GoalState:
    """Goal state G_t."""
    current_goal: Optional[str] = None
    subgoals: List[str] = field(default_factory=list)


@dataclass
class AgentState:
    """Complete agent state S_t = (P_t, M_t, B_t, G_t)."""
    control_plane: ControlPlane = field(default_factory=ControlPlane)
    memory: MemoryState = field(default_factory=MemoryState)
    beliefs: BeliefState = field(default_factory=BeliefState)
    goals: GoalState = field(default_factory=GoalState)
    ir_nodes: List[IRNode] = field(default_factory=list)
    step: int = 0

    def add_ir_node(self, node: IRNode) -> None:
        self.ir_nodes.append(node)

    def get_untainted_nodes(self) -> List[IRNode]:
        return [node for node in self.ir_nodes if node.is_untainted()]

    def get_trusted_nodes(self) -> List[IRNode]:
        return [node for node in self.ir_nodes if node.is_trusted()]

    def increment_step(self) -> None:
        self.step += 1

    def get_control_plane_hash(self) -> str:
        return self.control_plane.get_state_hash()


@dataclass
class ControlPlaneProposal:
    """Proposal for control-plane update.

    Generated by RLM controller, includes proposed changes and certificate.
    """
    proposal_id: str = field(
        default_factory=lambda: hashlib.sha256(str(datetime.now().timestamp()).encode()).hexdigest()[:16]
    )
    change_type: str = ""
    change_details: Dict[str, Any] = field(default_factory=dict)
    certificate: Optional[JustificationCertificate] = None
    requires_user_confirmation: bool = True

    def verify_certificate(self) -> bool:
        if self.certificate is None:
            return False
        return self.certificate.verify()
