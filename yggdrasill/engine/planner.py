from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

_plan_cache: Dict[Tuple[int, int], List[Tuple[str, Any]]] = {}


def build_plan(structure: Any) -> List[Tuple[str, Any]]:
    """Build an execution plan for the given structure.

    Returns a list of steps:
      ("node", node_id)               -- execute one node once
      ("cycle", (rep, frozenset(ids))) -- execute nodes K times (K from options at run-time)
    """
    cache_key = (id(structure), structure.execution_version)
    if cache_key in _plan_cache:
        return _plan_cache[cache_key]

    node_ids = sorted(structure.node_ids)
    edges = structure.get_edges()

    adj: Dict[str, Set[str]] = {nid: set() for nid in node_ids}
    for edge in edges:
        if edge.source_node in adj and edge.target_node in adj:
            adj[edge.source_node].add(edge.target_node)

    sccs = _tarjan(node_ids, adj)

    scc_order = _topo_sort_sccs(sccs, adj)

    has_self_loop = set()
    for edge in edges:
        if edge.source_node == edge.target_node:
            has_self_loop.add(edge.source_node)

    plan: List[Tuple[str, Any]] = []
    for comp in scc_order:
        if len(comp) == 1:
            nid = next(iter(comp))
            if nid in has_self_loop:
                plan.append(("cycle", (nid, frozenset(comp))))
            else:
                plan.append(("node", nid))
        else:
            rep = sorted(comp)[0]
            plan.append(("cycle", (rep, frozenset(comp))))

    _plan_cache[cache_key] = plan
    return plan


def clear_plan_cache() -> None:
    _plan_cache.clear()


# ---------------------------------------------------------------------------
# Tarjan's SCC algorithm
# ---------------------------------------------------------------------------

def _tarjan(node_ids: List[str], adj: Dict[str, Set[str]]) -> List[Set[str]]:
    index_counter = [0]
    stack: List[str] = []
    on_stack: Set[str] = set()
    index: Dict[str, int] = {}
    lowlink: Dict[str, int] = {}
    result: List[Set[str]] = []

    def strongconnect(v: str) -> None:
        index[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)

        for w in adj.get(v, ()):
            if w not in index:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], index[w])

        if lowlink[v] == index[v]:
            comp: Set[str] = set()
            while True:
                w = stack.pop()
                on_stack.discard(w)
                comp.add(w)
                if w == v:
                    break
            result.append(comp)

    for v in node_ids:
        if v not in index:
            strongconnect(v)

    return result


def _topo_sort_sccs(
    sccs: List[Set[str]], adj: Dict[str, Set[str]],
) -> List[Set[str]]:
    """Topologically sort SCCs (sources first)."""
    scc_of: Dict[str, int] = {}
    for i, comp in enumerate(sccs):
        for nid in comp:
            scc_of[nid] = i

    n = len(sccs)
    scc_adj: Dict[int, Set[int]] = {i: set() for i in range(n)}
    in_deg: Dict[int, int] = {i: 0 for i in range(n)}

    for src, targets in adj.items():
        si = scc_of.get(src)
        if si is None:
            continue
        for tgt in targets:
            ti = scc_of.get(tgt)
            if ti is None or ti == si:
                continue
            if ti not in scc_adj[si]:
                scc_adj[si].add(ti)
                in_deg[ti] += 1

    queue = sorted([i for i in range(n) if in_deg[i] == 0])
    order: List[int] = []
    while queue:
        cur = queue.pop(0)
        order.append(cur)
        for nxt in sorted(scc_adj[cur]):
            in_deg[nxt] -= 1
            if in_deg[nxt] == 0:
                queue.append(nxt)
                queue.sort()

    return [sccs[i] for i in order]
