"""
Control-Plane Integrity Verifier

Implements the verification logic from the theorem:
- Checks that control-plane updates are justified by untainted IR nodes
- Enforces that effective principal is SYS or confirmed USER
- Requires explicit user confirmation for integration/permission changes
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Callable, List, Dict, Any

from .agent_state import (
    AgentState,
    ControlPlaneProposal,
    JustificationCertificate,
    Principal,
    TaintStatus,
    IRNode,
    Permission,
    Integration,
    Policy,
)

logger = logging.getLogger(__name__)


class ControlPlaneVerifier:
    """Verifier that enforces control-plane integrity.

    According to theorem requirements:
    1. All nodes in dependency set must be untainted (tau = 0)
    2. Effective principal must be SYS or confirmed USER
    3. New integrations/permissions require explicit user confirmation
    """

    def __init__(
        self,
        user_confirmation_callback: Optional[Callable[[ControlPlaneProposal], bool]] = None,
    ):
        self.user_confirmation_callback = user_confirmation_callback
        self.verification_log: List[Dict[str, Any]] = []

    def verify_proposal(
        self, proposal: ControlPlaneProposal, state: AgentState
    ) -> Tuple[bool, str]:
        """Verify a control-plane update proposal.

        Returns:
            Tuple of (approved, reason).
        """
        logger.info("Verifying proposal %s: %s", proposal.proposal_id, proposal.change_type)

        # Step 1: Check if certificate exists
        if proposal.certificate is None:
            reason = "Proposal rejected: No justification certificate provided"
            logger.warning(reason)
            self._log_verification(proposal, False, reason)
            return False, reason

        # Step 2: Verify certificate
        if not proposal.certificate.verify():
            reason = "Proposal rejected: Certificate verification failed"
            logger.warning(reason)
            self._log_verification(proposal, False, reason)
            return False, reason

        # Step 3: Check all dependency nodes are untainted
        tainted_nodes = [
            node for node in proposal.certificate.dependency_nodes
            if not node.is_untainted()
        ]
        if tainted_nodes:
            reason = f"Proposal rejected: {len(tainted_nodes)} tainted nodes in dependency set"
            logger.warning("%s: %s", reason, [node.node_id for node in tainted_nodes])
            self._log_verification(proposal, False, reason)
            return False, reason

        # Step 4: Check effective principal is trusted
        if proposal.certificate.effective_principal not in (Principal.SYS, Principal.USER):
            reason = (
                f"Proposal rejected: Untrusted principal "
                f"{proposal.certificate.effective_principal.value}"
            )
            logger.warning(reason)
            self._log_verification(proposal, False, reason)
            return False, reason

        # Step 5: Check if user confirmation is required
        if proposal.requires_user_confirmation:
            if self.user_confirmation_callback is None:
                reason = "Proposal rejected: User confirmation required but no callback provided"
                logger.warning(reason)
                self._log_verification(proposal, False, reason)
                return False, reason

            user_approved = self.user_confirmation_callback(proposal)
            if not user_approved:
                reason = "Proposal rejected: User denied confirmation"
                logger.info(reason)
                self._log_verification(proposal, False, reason)
                return False, reason

            logger.info("Proposal approved by user")

        # All checks passed
        reason = "Proposal approved: All verification checks passed"
        logger.info(reason)
        self._log_verification(proposal, True, reason)
        return True, reason

    def apply_proposal(self, proposal: ControlPlaneProposal, state: AgentState) -> bool:
        """Apply an approved proposal to the agent state."""
        try:
            if proposal.change_type == "add_permission":
                permission = Permission(**proposal.change_details)
                state.control_plane.add_permission(permission)
                logger.info("Added permission: %s:%s", permission.tool_name, permission.action)
                return True

            elif proposal.change_type == "remove_permission":
                permission = Permission(**proposal.change_details)
                state.control_plane.remove_permission(permission)
                logger.info("Removed permission: %s:%s", permission.tool_name, permission.action)
                return True

            elif proposal.change_type == "add_integration":
                integration = Integration(**proposal.change_details)
                state.control_plane.add_integration(integration)
                logger.info(
                    "Added integration: %s:%s",
                    integration.integration_type, integration.name,
                )
                return True

            elif proposal.change_type == "remove_integration":
                integration = Integration(**proposal.change_details)
                state.control_plane.remove_integration(integration)
                logger.info(
                    "Removed integration: %s:%s",
                    integration.integration_type, integration.name,
                )
                return True

            elif proposal.change_type == "add_policy":
                policy = Policy(**proposal.change_details)
                state.control_plane.add_policy(policy)
                logger.info("Added policy: %s:%s", policy.principal.value, policy.resource)
                return True

            else:
                logger.error("Unknown change type: %s", proposal.change_type)
                return False

        except Exception as e:
            logger.error("Failed to apply proposal: %s", e)
            return False

    def verify_and_apply(
        self, proposal: ControlPlaneProposal, state: AgentState
    ) -> Tuple[bool, str]:
        """Verify and apply a proposal in one step."""
        approved, reason = self.verify_proposal(proposal, state)
        if not approved:
            return False, reason

        success = self.apply_proposal(proposal, state)
        if not success:
            return False, "Failed to apply approved proposal"

        return True, "Proposal verified and applied successfully"

    def _log_verification(
        self, proposal: ControlPlaneProposal, approved: bool, reason: str
    ) -> None:
        self.verification_log.append({
            "proposal_id": proposal.proposal_id,
            "change_type": proposal.change_type,
            "approved": approved,
            "reason": reason,
            "timestamp": (
                proposal.certificate.timestamp.isoformat()
                if proposal.certificate else None
            ),
        })

    def get_verification_stats(self) -> Dict[str, Any]:
        total = len(self.verification_log)
        approved = sum(1 for v in self.verification_log if v["approved"])
        rejected = total - approved
        return {
            "total_proposals": total,
            "approved": approved,
            "rejected": rejected,
            "approval_rate": approved / total if total > 0 else 0.0,
        }


class BaselineVerifier:
    """Baseline verifier with no control-plane integrity checks.

    This is the vulnerable baseline that accepts all proposals.
    Used for comparison in experiments.
    """

    def __init__(self):
        self.verification_log: List[Dict[str, Any]] = []

    def verify_proposal(
        self, proposal: ControlPlaneProposal, state: AgentState
    ) -> Tuple[bool, str]:
        """Always approve proposals (vulnerable baseline)."""
        reason = "Proposal approved: Baseline verifier accepts all proposals"
        logger.info("Baseline verifier approving proposal %s", proposal.proposal_id)
        self._log_verification(proposal, True, reason)
        return True, reason

    def apply_proposal(self, proposal: ControlPlaneProposal, state: AgentState) -> bool:
        try:
            if proposal.change_type == "add_permission":
                permission = Permission(**proposal.change_details)
                state.control_plane.add_permission(permission)
                return True

            elif proposal.change_type == "add_integration":
                integration = Integration(**proposal.change_details)
                state.control_plane.add_integration(integration)
                return True

            elif proposal.change_type == "add_policy":
                policy = Policy(**proposal.change_details)
                state.control_plane.add_policy(policy)
                return True

            else:
                return False

        except Exception as e:
            logger.error("Failed to apply proposal: %s", e)
            return False

    def verify_and_apply(
        self, proposal: ControlPlaneProposal, state: AgentState
    ) -> Tuple[bool, str]:
        approved, reason = self.verify_proposal(proposal, state)
        if not approved:
            return False, reason

        success = self.apply_proposal(proposal, state)
        if not success:
            return False, "Failed to apply proposal"

        return True, "Proposal applied successfully"

    def _log_verification(
        self, proposal: ControlPlaneProposal, approved: bool, reason: str
    ) -> None:
        self.verification_log.append({
            "proposal_id": proposal.proposal_id,
            "change_type": proposal.change_type,
            "approved": approved,
            "reason": reason,
        })

    def get_verification_stats(self) -> Dict[str, Any]:
        total = len(self.verification_log)
        approved = sum(1 for v in self.verification_log if v["approved"])
        rejected = total - approved
        return {
            "total_proposals": total,
            "approved": approved,
            "rejected": rejected,
            "approval_rate": approved / total if total > 0 else 0.0,
        }
