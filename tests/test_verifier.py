"""Tests for the control-plane verifier."""

import pytest

from src.ir import IRGraph
from src.principals import Principal
from src.state import ControlPlane, Integration, Permission, Policy
from src.verifier import (
    ControlPlaneProposal,
    ControlPlaneVerifier,
    JustificationCertificate,
    Verdict,
)


def _make_verifier():
    return ControlPlaneVerifier()


class TestVerifierBasics:
    def test_no_change_is_approved(self):
        """If proposed state equals current state, approve trivially."""
        v = _make_verifier()
        g = IRGraph()
        cp = ControlPlane()
        cert = JustificationCertificate(frozenset())
        proposal = ControlPlaneProposal(proposed_state=cp, certificate=cert)
        result = v.verify(proposal, g, cp)
        assert result.approved

    def test_trusted_justification_approved(self):
        """A change justified by a trusted, untainted node is approved."""
        v = _make_verifier()
        g = IRGraph()
        node = g.add_node(Principal.USER, "user says add tool")
        current = ControlPlane()
        proposed = ControlPlane(
            permissions={Permission("new_tool", frozenset({"read"}))},
        )
        cert = JustificationCertificate(
            justifying_node_ids=frozenset({node.id}),
            user_confirmed=True,
        )
        proposal = ControlPlaneProposal(
            proposed_state=proposed,
            certificate=cert,
            changes_permissions=True,
        )
        result = v.verify(proposal, g, current)
        assert result.approved


class TestVerifierRejectsUntrusted:
    def test_web_principal_rejected(self):
        """A change justified by a WEB node is rejected."""
        v = _make_verifier()
        g = IRGraph()
        node = g.add_node(Principal.WEB, "add telegram bot")
        current = ControlPlane()
        proposed = ControlPlane(
            integrations={Integration("telegram", "chat_channel")},
        )
        cert = JustificationCertificate(
            justifying_node_ids=frozenset({node.id}),
            user_confirmed=False,
        )
        proposal = ControlPlaneProposal(
            proposed_state=proposed,
            certificate=cert,
            adds_integration=True,
        )
        result = v.verify(proposal, g, current)
        assert not result.approved
        assert any("tainted" in r.lower() or "untrusted" in r.lower()
                    for r in result.reasons)

    def test_skill_principal_rejected(self):
        """A change justified by a SKILL node is rejected."""
        v = _make_verifier()
        g = IRGraph()
        node = g.add_node(Principal.SKILL, "grant admin")
        current = ControlPlane()
        proposed = ControlPlane(
            permissions={Permission("admin", frozenset({"all"}))},
        )
        cert = JustificationCertificate(
            justifying_node_ids=frozenset({node.id}),
            user_confirmed=False,
        )
        proposal = ControlPlaneProposal(
            proposed_state=proposed,
            certificate=cert,
            changes_permissions=True,
        )
        result = v.verify(proposal, g, current)
        assert not result.approved

    def test_tool_output_principal_rejected(self):
        """A change justified by a TOOL_OUTPUT node is rejected."""
        v = _make_verifier()
        g = IRGraph()
        node = g.add_node(Principal.TOOL_OUTPUT, "elevate permissions")
        current = ControlPlane()
        proposed = ControlPlane(
            policies={Policy(Principal.TOOL_OUTPUT, frozenset({"/"}))},
        )
        cert = JustificationCertificate(
            justifying_node_ids=frozenset({node.id}),
            user_confirmed=False,
        )
        proposal = ControlPlaneProposal(
            proposed_state=proposed,
            certificate=cert,
            changes_policies=True,
        )
        result = v.verify(proposal, g, current)
        assert not result.approved


class TestVerifierTransitiveTaint:
    def test_tainted_dependency_chain_rejected(self):
        """Even if the justifying node is USER, tainted deps cause rejection."""
        v = _make_verifier()
        g = IRGraph()
        evil = g.add_node(Principal.SKILL, "malicious instruction")
        derived = g.add_node(
            Principal.USER,
            "user unknowingly relays",
            dependencies=frozenset({evil.id}),
        )
        current = ControlPlane()
        proposed = ControlPlane(
            integrations={Integration("backdoor", "api_connection")},
        )
        cert = JustificationCertificate(
            justifying_node_ids=frozenset({derived.id}),
            user_confirmed=True,
        )
        proposal = ControlPlaneProposal(
            proposed_state=proposed,
            certificate=cert,
            adds_integration=True,
        )
        result = v.verify(proposal, g, current)
        assert not result.approved
        assert any("tainted" in r.lower() for r in result.reasons)


class TestVerifierUserConfirmation:
    def test_integration_without_confirmation_rejected(self):
        """Adding an integration without user confirmation is rejected."""
        v = _make_verifier()
        g = IRGraph()
        node = g.add_node(Principal.USER, "add slack")
        current = ControlPlane()
        proposed = ControlPlane(
            integrations={Integration("slack", "chat_channel")},
        )
        cert = JustificationCertificate(
            justifying_node_ids=frozenset({node.id}),
            user_confirmed=False,  # Missing confirmation
        )
        proposal = ControlPlaneProposal(
            proposed_state=proposed,
            certificate=cert,
            adds_integration=True,
        )
        result = v.verify(proposal, g, current)
        assert not result.approved
        assert any("confirmation" in r.lower() for r in result.reasons)

    def test_integration_with_confirmation_approved(self):
        """Adding an integration with user confirmation is approved."""
        v = _make_verifier()
        g = IRGraph()
        node = g.add_node(Principal.USER, "add slack")
        current = ControlPlane()
        proposed = ControlPlane(
            integrations={Integration("slack", "chat_channel")},
        )
        cert = JustificationCertificate(
            justifying_node_ids=frozenset({node.id}),
            user_confirmed=True,
        )
        proposal = ControlPlaneProposal(
            proposed_state=proposed,
            certificate=cert,
            adds_integration=True,
        )
        result = v.verify(proposal, g, current)
        assert result.approved
