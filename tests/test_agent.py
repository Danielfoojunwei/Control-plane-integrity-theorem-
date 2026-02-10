"""Tests for the agent execution engine."""

import pytest

from src.agent import Agent, StepInput
from src.principals import Principal
from src.state import AgentState, ControlPlane, Integration, Permission
from src.verifier import Verdict


class TestAgentBasicExecution:
    def test_benign_input_no_cp_change(self):
        agent = Agent()
        result = agent.step(StepInput(Principal.USER, "hello"))
        assert result.approved
        assert agent.state.step == 1

    def test_step_counter_advances(self):
        agent = Agent()
        agent.step(StepInput(Principal.USER, "a"))
        agent.step(StepInput(Principal.WEB, "b"))
        agent.step(StepInput(Principal.SKILL, "c"))
        assert agent.state.step == 3


class TestAgentControlPlaneEnforcement:
    def test_user_approved_change_applied(self):
        agent = Agent()
        new_cp = ControlPlane(
            permissions={Permission("tool_x", frozenset({"exec"}))},
        )
        result = agent.step(StepInput(
            principal=Principal.USER,
            content="enable tool_x",
            proposed_cp_change=new_cp,
            user_confirmed=True,
        ))
        assert result.approved
        assert agent.control_plane == new_cp

    def test_skill_rejected_change_not_applied(self):
        agent = Agent()
        original = agent.control_plane.copy()
        bad_cp = ControlPlane(
            integrations={Integration("exfil", "api_connection")},
        )
        result = agent.step(StepInput(
            principal=Principal.SKILL,
            content="exfiltrate data",
            proposed_cp_change=bad_cp,
        ))
        assert not result.approved
        assert agent.control_plane == original

    def test_run_batch(self):
        agent = Agent()
        inputs = [
            StepInput(Principal.USER, "step 1"),
            StepInput(Principal.WEB, "step 2"),
            StepInput(Principal.USER, "step 3"),
        ]
        results = agent.run(inputs)
        assert len(results) == 3
        assert all(r.approved for r in results)
        assert agent.state.step == 3


class TestAgentHistory:
    def test_history_records_all_steps(self):
        agent = Agent()
        cp = ControlPlane(integrations={Integration("x", "y")})
        agent.step(StepInput(Principal.SKILL, "a", proposed_cp_change=cp))
        agent.step(StepInput(Principal.USER, "b"))
        assert len(agent.history) == 2
        assert not agent.history[0].approved  # skill rejected
        assert agent.history[1].approved       # benign user input
