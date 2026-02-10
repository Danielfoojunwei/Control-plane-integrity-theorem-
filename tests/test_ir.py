"""Tests for the Information Representation graph and taint propagation."""

import pytest

from src.ir import IRGraph, IRNode
from src.principals import Principal


class TestIRNodeCreation:
    def test_trusted_node_is_untainted(self):
        g = IRGraph()
        node = g.add_node(Principal.SYS, "system policy")
        assert node.taint == 0

    def test_user_node_is_untainted(self):
        g = IRGraph()
        node = g.add_node(Principal.USER, "user command")
        assert node.taint == 0

    def test_web_node_is_tainted(self):
        g = IRGraph()
        node = g.add_node(Principal.WEB, "scraped page")
        assert node.taint == 1

    def test_skill_node_is_tainted(self):
        g = IRGraph()
        node = g.add_node(Principal.SKILL, "skill content")
        assert node.taint == 1

    def test_tool_output_node_is_tainted(self):
        g = IRGraph()
        node = g.add_node(Principal.TOOL_OUTPUT, "tool result")
        assert node.taint == 1


class TestTaintPropagation:
    def test_derived_from_tainted_is_tainted(self):
        """A trusted node derived from a tainted dependency becomes tainted."""
        g = IRGraph()
        tainted = g.add_node(Principal.WEB, "malicious page")
        derived = g.add_node(
            Principal.USER,
            "user sees summary of page",
            dependencies=frozenset({tainted.id}),
        )
        assert derived.taint == 1

    def test_derived_from_untainted_is_untainted(self):
        g = IRGraph()
        clean = g.add_node(Principal.SYS, "system config")
        derived = g.add_node(
            Principal.USER,
            "user-approved change",
            dependencies=frozenset({clean.id}),
        )
        assert derived.taint == 0

    def test_mixed_dependencies_yields_tainted(self):
        """If any dependency is tainted, the derived node is tainted."""
        g = IRGraph()
        clean = g.add_node(Principal.SYS, "clean")
        dirty = g.add_node(Principal.SKILL, "malicious skill")
        derived = g.add_node(
            Principal.USER,
            "mix",
            dependencies=frozenset({clean.id, dirty.id}),
        )
        assert derived.taint == 1

    def test_transitive_taint(self):
        """Taint propagates through chains of derivation."""
        g = IRGraph()
        root = g.add_node(Principal.WEB, "evil")
        mid = g.add_node(Principal.SYS, "relay", dependencies=frozenset({root.id}))
        leaf = g.add_node(Principal.USER, "final", dependencies=frozenset({mid.id}))
        assert root.taint == 1
        assert mid.taint == 1
        assert leaf.taint == 1


class TestDependencySet:
    def test_transitive_closure(self):
        g = IRGraph()
        a = g.add_node(Principal.SYS, "a")
        b = g.add_node(Principal.SYS, "b", dependencies=frozenset({a.id}))
        c = g.add_node(Principal.SYS, "c", dependencies=frozenset({b.id}))
        deps = g.get_dependency_set(c.id)
        assert deps == {a.id, b.id, c.id}

    def test_missing_dependency_raises(self):
        g = IRGraph()
        with pytest.raises(KeyError):
            g.add_node(Principal.SYS, "bad", dependencies=frozenset({9999}))
