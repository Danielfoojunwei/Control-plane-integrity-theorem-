"""
Control-plane update verifier.

Implements the three-phase update protocol:
  1. Proposal  -- the RLM controller proposes P_{t+1} and a tool call.
  2. Certification -- the proposal includes a justification certificate
     listing the IR nodes that justify the change.
  3. Verification -- the verifier checks that every node in the
     dependency set is untainted (tau=0) and has trusted provenance.

The verifier is the core enforcement mechanism for the Control-Plane
Integrity Theorem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import FrozenSet, List, Optional, Set

from .ir import IRGraph, IRNode
from .principals import Principal, is_trusted_principal
from .state import ControlPlane


# ── Verification result ──────────────────────────────────────────────────

class Verdict(Enum):
    APPROVED = auto()
    REJECTED = auto()


@dataclass(frozen=True)
class VerificationResult:
    verdict: Verdict
    reasons: tuple  # tuple of strings explaining rejection reasons

    @property
    def approved(self) -> bool:
        return self.verdict == Verdict.APPROVED


# ── Justification certificate ────────────────────────────────────────────

@dataclass(frozen=True)
class JustificationCertificate:
    """Certificate attached to a control-plane update proposal.

    Fields:
        justifying_node_ids: IR node ids that the proposer claims
            justify the change.
        user_confirmed: whether explicit user confirmation was obtained
            for integration or permission changes.
    """
    justifying_node_ids: FrozenSet[int]
    user_confirmed: bool = False


# ── Control-plane update proposal ────────────────────────────────────────

@dataclass(frozen=True)
class ControlPlaneProposal:
    """A proposed update to the control plane.

    Fields:
        proposed_state: the new control-plane state P_{t+1}.
        certificate: justification for the change.
        adds_integration: True if the proposal adds a new integration.
        changes_permissions: True if the proposal modifies permissions.
        changes_policies: True if the proposal modifies policies.
    """
    proposed_state: ControlPlane
    certificate: JustificationCertificate
    adds_integration: bool = False
    changes_permissions: bool = False
    changes_policies: bool = False


# ── Verifier ─────────────────────────────────────────────────────────────

class ControlPlaneVerifier:
    """Stateless verifier that enforces control-plane integrity.

    Verification rules:
      V1. All nodes in the transitive dependency set of the justifying
          nodes must have taint == 0.
      V2. The principal of every justifying node must be SYS or USER.
      V3. If the proposal adds integrations or changes permissions,
          explicit user confirmation must be present in the certificate.
    """

    def verify(
        self,
        proposal: ControlPlaneProposal,
        ir_graph: IRGraph,
        current: ControlPlane,
    ) -> VerificationResult:
        """Verify a control-plane update proposal.

        Returns an APPROVED result only if all verification rules pass.
        """
        reasons: List[str] = []

        # If nothing changed, approve trivially.
        if proposal.proposed_state == current:
            return VerificationResult(Verdict.APPROVED, ())

        cert = proposal.certificate

        # Collect all nodes in the transitive dependency set.
        all_dep_ids: Set[int] = set()
        for nid in cert.justifying_node_ids:
            if nid not in ir_graph:
                reasons.append(
                    f"Justifying node {nid} does not exist in the IR graph."
                )
                continue
            all_dep_ids |= ir_graph.get_dependency_set(nid)

        if reasons:
            return VerificationResult(Verdict.REJECTED, tuple(reasons))

        # V1: All nodes in the dependency set must be untainted.
        for nid in all_dep_ids:
            node = ir_graph[nid]
            if node.taint != 0:
                reasons.append(
                    f"Node {nid} (principal={node.principal.name}) is tainted "
                    f"(tau={node.taint})."
                )

        # V2: All justifying nodes must have trusted principal.
        for nid in cert.justifying_node_ids:
            node = ir_graph[nid]
            if not is_trusted_principal(node.principal):
                reasons.append(
                    f"Justifying node {nid} has untrusted principal "
                    f"{node.principal.name}."
                )

        # V3: Integration/permission changes require user confirmation.
        needs_confirmation = (
            proposal.adds_integration
            or proposal.changes_permissions
            or proposal.changes_policies
        )
        if needs_confirmation and not cert.user_confirmed:
            reasons.append(
                "Proposal modifies integrations, permissions, or policies "
                "but lacks explicit user confirmation."
            )

        if reasons:
            return VerificationResult(Verdict.REJECTED, tuple(reasons))

        return VerificationResult(Verdict.APPROVED, ())
