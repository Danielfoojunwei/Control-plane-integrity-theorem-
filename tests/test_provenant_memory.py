"""Tests for provenance-preserving memory store."""

import pytest
from src.provenant_memory import ProvenantMemoryStore, MemoryEntry
from src.ir import IRGraph
from src.principals import Principal


@pytest.fixture
def setup():
    """Create a fresh IR graph and memory store."""
    g = IRGraph()
    mem = ProvenantMemoryStore()
    return g, mem


class TestProvenancePreservation:
    """Test that taint propagates through memory store/load cycles."""

    def test_tainted_content_stays_tainted_through_memory(self, setup):
        """Content from untrusted source, stored then retrieved, must stay tainted."""
        g, mem = setup
        # Untrusted content enters
        src_node = g.add_node(Principal.SKILL, "malicious payload")
        assert src_node.taint == 1

        # Store in memory
        mem.store("notes", "malicious payload", src_node.id, step=0, ir_graph=g)

        # Retrieve — the new node MUST be tainted via dependency chain
        retrieved = mem.retrieve("notes", g)
        assert retrieved is not None
        assert retrieved.taint == 1  # THIS IS THE KEY TEST

    def test_clean_content_stays_clean_through_memory(self, setup):
        """Content from trusted source, stored then retrieved, stays clean."""
        g, mem = setup
        src_node = g.add_node(Principal.USER, "user note")
        assert src_node.taint == 0

        mem.store("notes", "user note", src_node.id, step=0, ir_graph=g)
        retrieved = mem.retrieve("notes", g)
        assert retrieved is not None
        assert retrieved.taint == 0

    def test_overwritten_entry_preserves_latest_provenance(self, setup):
        """When content is overwritten, provenance tracks the latest write."""
        g, mem = setup

        # First write: clean
        clean_node = g.add_node(Principal.USER, "clean data")
        mem.store("key", "clean data", clean_node.id, step=0, ir_graph=g)

        # Overwrite with tainted content
        tainted_node = g.add_node(Principal.WEB, "tainted data")
        mem.store("key", "tainted data", tainted_node.id, step=1, ir_graph=g)

        # Retrieve should now be tainted
        retrieved = mem.retrieve("key", g)
        assert retrieved is not None
        assert retrieved.taint == 1

    def test_write_history_preserved(self, setup):
        """Full write history is maintained for audit."""
        g, mem = setup
        n1 = g.add_node(Principal.USER, "v1")
        n2 = g.add_node(Principal.USER, "v2")
        mem.store("key", "v1", n1.id, step=0, ir_graph=g)
        mem.store("key", "v2", n2.id, step=1, ir_graph=g)

        history = mem.get_write_history("key")
        assert len(history) == 2


class TestMemoryLaunderingPrevention:
    """Test that the provenant memory store prevents taint laundering."""

    def test_laundering_blocked(self, setup):
        """The exact scenario from Part B2 Scenario 4: store and retrieve
        should NOT launder taint away."""
        g, mem = setup

        # Step 1: Untrusted content arrives
        attack_node = g.add_node(Principal.SKILL, "add telegram bot")
        assert attack_node.taint == 1

        # Step 2: LLM processes it (derived, tainted)
        llm_node = g.add_node(
            Principal.SYS, "tool_call: add_integration",
            frozenset({attack_node.id})
        )
        assert llm_node.taint == 1

        # Step 3: Store LLM output in memory
        mem.store("reminder", "add telegram bot", llm_node.id, step=0, ir_graph=g)

        # Step 4: Later, retrieve from memory
        retrieved = mem.retrieve("reminder", g)

        # Without provenant memory, this would create a fresh SYS node
        # with taint=0 (laundered). With provenant memory, taint=1 is preserved.
        assert retrieved is not None
        assert retrieved.taint == 1

    def test_multi_hop_laundering_blocked(self, setup):
        """Even multiple store/load cycles preserve taint."""
        g, mem = setup

        # Tainted origin
        src = g.add_node(Principal.WEB, "inject")

        # Store → retrieve → store → retrieve
        mem.store("k1", "inject", src.id, step=0, ir_graph=g)
        r1 = mem.retrieve("k1", g)
        assert r1.taint == 1

        mem.store("k2", "inject_copy", r1.id, step=1, ir_graph=g)
        r2 = mem.retrieve("k2", g)
        assert r2.taint == 1  # Still tainted after 2 hops through memory


class TestMemoryStoreBasics:
    """Test basic memory store operations."""

    def test_missing_key_returns_none(self, setup):
        g, mem = setup
        assert mem.retrieve("nonexistent", g) is None

    def test_has_key(self, setup):
        g, mem = setup
        n = g.add_node(Principal.USER, "data")
        mem.store("key", "data", n.id, step=0, ir_graph=g)
        assert mem.has_key("key")
        assert not mem.has_key("other")

    def test_invalid_source_node_raises(self, setup):
        g, mem = setup
        with pytest.raises(KeyError):
            mem.store("key", "data", 99999, step=0, ir_graph=g)

    def test_keys_property(self, setup):
        g, mem = setup
        n = g.add_node(Principal.USER, "data")
        mem.store("a", "data", n.id, step=0, ir_graph=g)
        mem.store("b", "data", n.id, step=0, ir_graph=g)
        assert mem.keys == {"a", "b"}
