"""
Provenance-preserving memory store.

Addresses Reviewer Concern #12 (memory laundering): When content is stored
in agent memory and later retrieved, the provenance chain must be preserved.
Without this, an attacker can launder taint by storing untrusted content in
memory and retrieving it as a fresh, untainted SYS node.

Design:
  - Every memory entry stores both content AND the IR node ID that produced it.
  - On retrieval, the returned IR node carries a dependency on the original
    node ID, preserving the taint chain.
  - The memory store itself is append-only for provenance records; content
    can be overwritten but the provenance link to the latest write is retained.

This closes Part B2 Scenario 4 (taint laundering via memory).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Set

from .ir import IRGraph, IRNode
from .principals import Principal


@dataclass(frozen=True)
class MemoryEntry:
    """A single memory entry with provenance tracking."""
    key: str
    content: str
    source_node_id: int          # IR node that caused this write
    write_step: int              # Agent step when written


class ProvenantMemoryStore:
    """Memory store that preserves provenance through store/load cycles.

    The key invariant:
        If content with taint=1 is stored at key K, then any node
        created by reading key K will inherit taint=1 through the
        dependency chain.

    This is achieved by recording the source_node_id at write time
    and adding it as a dependency at read time.
    """

    def __init__(self) -> None:
        self._entries: Dict[str, MemoryEntry] = {}
        self._write_history: Dict[str, list] = {}  # key → [MemoryEntry, ...]

    def store(
        self,
        key: str,
        content: str,
        source_node_id: int,
        step: int,
        ir_graph: IRGraph,
    ) -> MemoryEntry:
        """Store content with provenance tracking.

        Args:
            key: Memory key (e.g., "SOUL.md", "notes/meeting")
            content: The content to store
            source_node_id: IR node ID that produced this content
            step: Current agent step number
            ir_graph: The IR graph (used to validate node exists)

        Returns:
            The created MemoryEntry

        Raises:
            KeyError: If source_node_id doesn't exist in the IR graph
        """
        if source_node_id not in ir_graph:
            raise KeyError(
                f"Source node {source_node_id} not in IR graph. "
                "Memory writes must reference an existing IR node."
            )

        entry = MemoryEntry(
            key=key,
            content=content,
            source_node_id=source_node_id,
            write_step=step,
        )
        self._entries[key] = entry

        if key not in self._write_history:
            self._write_history[key] = []
        self._write_history[key].append(entry)

        return entry

    def retrieve(
        self,
        key: str,
        ir_graph: IRGraph,
        reader_principal: Principal = Principal.SYS,
    ) -> Optional[IRNode]:
        """Retrieve content from memory WITH provenance preservation.

        Creates a new IR node that carries a dependency on the node
        that originally wrote the content.  This ensures taint is
        propagated through memory store/load cycles.

        Args:
            key: Memory key to retrieve
            ir_graph: The IR graph to add the retrieval node to
            reader_principal: Principal performing the read (default SYS)

        Returns:
            A new IR node with the memory content and correct provenance,
            or None if the key doesn't exist.
        """
        entry = self._entries.get(key)
        if entry is None:
            return None

        # Create a new IR node that depends on the original write source.
        # This is the critical step: the dependency link preserves taint.
        retrieval_node = ir_graph.add_node(
            principal=reader_principal,
            content=entry.content,
            dependencies=frozenset({entry.source_node_id}),
        )
        return retrieval_node

    def has_key(self, key: str) -> bool:
        """Check if a key exists in memory."""
        return key in self._entries

    def get_source_node_id(self, key: str) -> Optional[int]:
        """Get the source node ID for a memory entry."""
        entry = self._entries.get(key)
        return entry.source_node_id if entry else None

    def get_write_history(self, key: str) -> list:
        """Get the full write history for a key."""
        return list(self._write_history.get(key, []))

    @property
    def keys(self) -> Set[str]:
        """All keys currently in memory."""
        return set(self._entries.keys())
