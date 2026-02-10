"""
Agent execution engine.

Models a stepwise agent that processes trusted and untrusted inputs,
maintains an IR graph, and gates control-plane mutations through the
verifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, List, Optional

from .ir import IRGraph, IRNode
from .principals import Principal
from .state import AgentState, ControlPlane
from .verifier import (
    ControlPlaneProposal,
    ControlPlaneVerifier,
    JustificationCertificate,
    VerificationResult,
    Verdict,
)


@dataclass
class StepInput:
    """An input arriving at the agent during one step.

    Fields:
        principal: the origin of this input.
        content: the payload string.
        proposed_cp_change: if non-None, the RLM proposes to update
            the control plane as part of processing this input.
        user_confirmed: whether explicit user confirmation accompanies
            this input (only meaningful for USER principal).
    """
    principal: Principal
    content: str
    proposed_cp_change: Optional[ControlPlane] = None
    user_confirmed: bool = False


class Agent:
    """Simulated agent with control-plane integrity enforcement.

    The agent processes a sequence of StepInputs.  Each step:
      1. Records the input as an IR node.
      2. If the input proposes a control-plane change, constructs a
         proposal with a justification certificate.
      3. Runs the verifier.  If rejected, the control plane is unchanged.
      4. Advances the step counter.
    """

    def __init__(self, initial_state: Optional[AgentState] = None) -> None:
        self.state = initial_state or AgentState()
        self._verifier = ControlPlaneVerifier()
        self._history: List[VerificationResult] = []

    @property
    def control_plane(self) -> ControlPlane:
        return self.state.control_plane

    @property
    def history(self) -> List[VerificationResult]:
        return list(self._history)

    def step(self, inp: StepInput) -> VerificationResult:
        """Process one input and return the verification result."""
        # 1. Record the input in the IR graph.
        ir_node = self.state.ir_graph.add_node(
            principal=inp.principal,
            content=inp.content,
        )

        # 2. If no control-plane change is proposed, approve trivially.
        if inp.proposed_cp_change is None:
            result = VerificationResult(Verdict.APPROVED, ())
            self._history.append(result)
            self.state.step += 1
            return result

        # 3. Build proposal and certificate.
        current_cp = self.state.control_plane
        proposed_cp = inp.proposed_cp_change

        adds_integration = (
            proposed_cp.integrations != current_cp.integrations
        )
        changes_permissions = (
            proposed_cp.permissions != current_cp.permissions
        )
        changes_policies = (
            proposed_cp.policies != current_cp.policies
        )

        certificate = JustificationCertificate(
            justifying_node_ids=frozenset({ir_node.id}),
            user_confirmed=inp.user_confirmed,
        )

        proposal = ControlPlaneProposal(
            proposed_state=proposed_cp,
            certificate=certificate,
            adds_integration=adds_integration,
            changes_permissions=changes_permissions,
            changes_policies=changes_policies,
        )

        # 4. Verify.
        result = self._verifier.verify(
            proposal, self.state.ir_graph, current_cp,
        )

        # 5. Apply if approved.
        if result.approved:
            self.state.control_plane = proposed_cp

        self._history.append(result)
        self.state.step += 1
        return result

    def run(self, inputs: List[StepInput]) -> List[VerificationResult]:
        """Run the agent over a sequence of inputs."""
        return [self.step(inp) for inp in inputs]
