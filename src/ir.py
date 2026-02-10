"""
Information Representation (IR) graph with taint tracking.

Every piece of data the agent processes is represented as an IR node.
Each node carries:
  - a unique identifier
  - a taint flag (tau): 0 = untainted, 1 = tainted
  - a principal: the origin of the information
  - content: the payload (opaque to the verifier)
  - dependencies: set of parent IR node ids from which this node was derived

Taint propagation rule:
  A derived node is tainted (tau=1) if ANY of its dependencies is tainted,
  or if its principal is untrusted.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Optional, Set

from .principals import Principal, is_trusted_principal


_ID_COUNTER = itertools.count()


def _next_id() -> int:
    return next(_ID_COUNTER)


@dataclass(frozen=True)
class IRNode:
    """An immutable node in the information-representation graph."""

    id: int
    principal: Principal
    taint: int                          # 0 = untainted, 1 = tainted
    content: str                        # opaque payload
    dependencies: FrozenSet[int] = frozenset()  # ids of parent nodes

    def __post_init__(self) -> None:
        if self.taint not in (0, 1):
            raise ValueError(f"taint must be 0 or 1, got {self.taint}")


class IRGraph:
    """Append-only graph of IR nodes with automatic taint propagation."""

    def __init__(self) -> None:
        self._nodes: Dict[int, IRNode] = {}

    # -- queries ----------------------------------------------------------

    def __getitem__(self, node_id: int) -> IRNode:
        return self._nodes[node_id]

    def __contains__(self, node_id: int) -> bool:
        return node_id in self._nodes

    def __len__(self) -> int:
        return len(self._nodes)

    @property
    def nodes(self) -> Dict[int, IRNode]:
        return dict(self._nodes)

    def get_dependency_set(self, node_id: int) -> Set[int]:
        """Return the transitive closure of dependencies for *node_id*."""
        visited: Set[int] = set()
        stack = [node_id]
        while stack:
            nid = stack.pop()
            if nid in visited:
                continue
            visited.add(nid)
            node = self._nodes[nid]
            stack.extend(node.dependencies)
        return visited

    # -- mutations --------------------------------------------------------

    def add_node(
        self,
        principal: Principal,
        content: str,
        dependencies: Optional[FrozenSet[int]] = None,
    ) -> IRNode:
        """Create a new IR node with automatic taint propagation.

        A node is tainted if:
          1. Its principal is untrusted, OR
          2. Any of its dependencies is tainted.
        """
        deps = dependencies or frozenset()
        # Check that all dependencies exist
        for dep_id in deps:
            if dep_id not in self._nodes:
                raise KeyError(f"dependency {dep_id} not in graph")

        # Compute taint
        taint = _compute_taint(principal, deps, self._nodes)

        node_id = _next_id()
        node = IRNode(
            id=node_id,
            principal=principal,
            taint=taint,
            content=content,
            dependencies=deps,
        )
        self._nodes[node_id] = node
        return node


def _compute_taint(
    principal: Principal,
    deps: FrozenSet[int],
    nodes: Dict[int, IRNode],
) -> int:
    """Determine taint for a new node.

    Tainted if the principal is untrusted or any dependency is tainted.
    """
    if not is_trusted_principal(principal):
        return 1
    for dep_id in deps:
        if nodes[dep_id].taint == 1:
            return 1
    return 0
