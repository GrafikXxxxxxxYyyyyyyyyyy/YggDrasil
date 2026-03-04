"""
Executor: topological order, SCC, buffer, run(graph, inputs, ...) -> outputs.

Canon: WorldGenerator_2.0/TODO_03_GRAPH_ENGINE.md §4.
Design: single entrypoint run() so pipeline (TODO_04) can call it per graph;
input/output contract by name for multi-endpoint (TODO_07).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from yggdrasill.foundation.graph import Edge, Graph

# Cache execution plan by (id(graph), graph._execution_version) to avoid recomputing (TODO_03 §4.1)
_execution_plan_cache: Dict[int, Tuple[int, List[Tuple[str, Any]]]] = {}


def _input_key(entry: Dict[str, Any]) -> str:
    """Stable key for inputs: name if set, else 'node_id:port_name'."""
    if entry.get("name") is not None:
        return str(entry["name"])
    return f"{entry['node_id']}:{entry['port_name']}"


def _output_key(entry: Dict[str, Any]) -> str:
    """Stable key for outputs (same convention as inputs)."""
    return _input_key(entry)


def _scc_tarjan(graph: Graph) -> List[Set[str]]:
    """
    Strongly connected components (Tarjan). Returns list of sets of node_ids.
    Order: reverse topological order of SCCs (sinks first).
    """
    node_ids = set(graph.node_ids)
    # Adjacency: node -> list of successors (out-edges: source -> target)
    succ: Dict[str, List[str]] = {nid: [] for nid in node_ids}
    for e in graph.get_edges():
        succ[e.source_node].append(e.target_node)

    index_counter = [0]
    stack: List[str] = []
    low: Dict[str, int] = {}
    index: Dict[str, int] = {}
    on_stack: Dict[str, bool] = {nid: False for nid in node_ids}
    sccs: List[Set[str]] = []

    def strongconnect(v: str) -> None:
        index[v] = index_counter[0]
        low[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack[v] = True
        for w in succ[v]:
            if w not in index:
                strongconnect(w)
                low[v] = min(low[v], low[w])
            elif on_stack[w]:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp: Set[str] = set()
            while True:
                w = stack.pop()
                on_stack[w] = False
                comp.add(w)
                if w == v:
                    break
            sccs.append(comp)

    for nid in node_ids:
        if nid not in index:
            strongconnect(nid)
    return sccs


def _topological_order_sccs(
    graph: Graph,
    sccs: List[Set[str]],
) -> List[Tuple[str, Set[str]]]:
    """
    Topological order of SCCs (SCC A before B if there is an edge from any node in A to any node in B).
    Returns list of (scc_id, set of node_ids) in execution order (sources first).
    scc_id is a representative node_id from the SCC for ordering.
    """
    # Map node -> index of its SCC
    node_to_scc_idx: Dict[str, int] = {}
    for i, comp in enumerate(sccs):
        for nid in comp:
            node_to_scc_idx[nid] = i
    # Build DAG of SCC indices: edge (i -> j) if edge from SCC i to SCC j (i != j)
    n_scc = len(sccs)
    scc_succ: List[Set[int]] = [set() for _ in range(n_scc)]
    for e in graph.get_edges():
        i = node_to_scc_idx[e.source_node]
        j = node_to_scc_idx[e.target_node]
        if i != j:
            scc_succ[i].add(j)
    # Kahn: in-degree for each SCC
    in_deg = [0] * n_scc
    for i in range(n_scc):
        for j in scc_succ[i]:
            in_deg[j] += 1
    queue = [i for i in range(n_scc) if in_deg[i] == 0]
    order: List[int] = []
    while queue:
        i = queue.pop(0)
        order.append(i)
        for j in scc_succ[i]:
            in_deg[j] -= 1
            if in_deg[j] == 0:
                queue.append(j)
    # If there are cycles in the SCC DAG we'd have missed nodes - but by construction we don't
    return [(min(sccs[i]), sccs[i]) for i in order]


def _build_execution_plan(graph: Graph) -> List[Tuple[str, Any]]:
    """
    Build execution plan: list of steps.
    Each step: ("node", node_id) for DAG node, or ("cycle", (scc_representative, set of node_ids)) for cycle.
    Cycles are SCCs with more than one node or one node with self-loop.
    """
    gid = id(graph)
    version = getattr(graph, "_execution_version", 0)
    if gid in _execution_plan_cache:
        cached_ver, cached_plan = _execution_plan_cache[gid]
        if cached_ver == version:
            return cached_plan
    sccs = _scc_tarjan(graph)
    # Self-loop: node has edge to itself
    self_loop: Set[str] = set()
    for e in graph.get_edges():
        if e.source_node == e.target_node:
            self_loop.add(e.source_node)
    ordered = _topological_order_sccs(graph, sccs)
    plan: List[Tuple[str, Any]] = []
    for _scc_rep, comp in ordered:
        if len(comp) > 1 or (len(comp) == 1 and next(iter(comp)) in self_loop):
            # Cycle: run all nodes in comp in deterministic order (sorted by node_id)
            plan.append(("cycle", (min(comp), comp)))
        else:
            # Singleton DAG node
            plan.append(("node", next(iter(comp))))
    _execution_plan_cache[gid] = (version, plan)
    return plan


def _gather_inputs_for_node(graph: Graph, node_id: str, buffer: Dict[Tuple[str, str], Any]) -> Dict[str, Any]:
    """Build inputs dict for block.forward from buffer: in-edges and pre-filled (node_id, port)."""
    inputs: Dict[str, Any] = {}
    for e in graph.get_edges_in(node_id):
        key = (e.source_node, e.source_port)
        if key in buffer:
            inputs[e.target_port] = buffer[key]
    node = graph.get_node(node_id)
    if node is not None:
        for port in node.block.get_input_ports():
            if port.name not in inputs:
                key = (node_id, port.name)
                if key in buffer:
                    inputs[port.name] = buffer[key]
                elif port.optional:
                    # Optional port: use default from block config if present (canon §4.1).
                    defaults = node.block.config.get("defaults") or {}
                    if port.name in defaults:
                        inputs[port.name] = defaults[port.name]
    return inputs


def _apply_outputs_to_buffer(
    node_id: str,
    outputs: Dict[str, Any],
    buffer: Dict[Tuple[str, str], Any],
) -> None:
    """Write block outputs to buffer by (node_id, port_name)."""
    for port_name, value in outputs.items():
        buffer[(node_id, port_name)] = value


def run(
    graph: Graph,
    inputs: Dict[str, Any],
    *,
    training: bool = False,
    num_loop_steps: Optional[int] = None,
    device: Optional[Any] = None,
    callbacks: Optional[List[Callable[..., None]]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Execute graph: run all nodes in topological order; run cycle subgraphs N times.

    - inputs: dict keyed by exposed input name (or "node_id:port_name" if no name).
    - training: set blocks to train mode if they have .train(mode).
    - num_loop_steps: for cycle subgraphs; default from graph.metadata["num_loop_steps"] or 1.
    - device: if blocks have .to(device), call it before execution.
    - callbacks: list of hook(node_id, phase, **kwargs). phase in ("before", "after", "loop_start", "loop_end").
    - dry_run: if True, do not call block.forward(); validate order and buffer keys, write None for outputs (canon §9.2).
    - returns: dict keyed by exposed output name (or "node_id:port_name").
    """
    # Resolve num_loop_steps
    n_loop = num_loop_steps
    if n_loop is None:
        n_loop = graph.metadata.get("num_loop_steps", 1)

    # Input spec -> buffer keys
    input_spec = graph.get_input_spec()
    key_to_node_port: Dict[str, Tuple[str, str]] = {}
    for entry in input_spec:
        key_to_node_port[_input_key(entry)] = (entry["node_id"], entry["port_name"])

    buffer: Dict[Tuple[str, str], Any] = {}
    exposed_input_keys = {(e["node_id"], e["port_name"]) for e in input_spec}
    for entry in input_spec:
        key = _input_key(entry)
        if key in inputs:
            buffer[(entry["node_id"], entry["port_name"])] = inputs[key]
    for k, v in inputs.items():
        if isinstance(k, tuple) and len(k) == 2:
            pt = (str(k[0]), str(k[1]))
            if pt in exposed_input_keys and pt not in buffer:
                buffer[pt] = v

    # Optional: device
    if device is not None:
        for nid in graph.node_ids:
            node = graph.get_node(nid)
            if node is not None and hasattr(node.block, "to") and callable(getattr(node.block, "to")):
                node.block.to(device)

    # Optional: training mode
    for nid in graph.node_ids:
        node = graph.get_node(nid)
        if node is None:
            continue
        if training and hasattr(node.block, "train"):
            node.block.train(True)
        elif not training and hasattr(node.block, "eval"):
            node.block.eval()

    plan = _build_execution_plan(graph)
    hooks = callbacks or []

    def _run_node(nid: str) -> None:
        for h in hooks:
            try:
                h(nid, "before", buffer=buffer)
            except Exception:
                pass
        node = graph.get_node(nid)
        if node is None:
            return
        inp = _gather_inputs_for_node(graph, nid, buffer)
        if dry_run:
            # Don't call forward; write None for each output port so downstream nodes see a value (canon §9.2).
            out = {p.name: None for p in node.block.get_output_ports()}
        else:
            out = node.block.forward(inp)
        _apply_outputs_to_buffer(nid, out, buffer)
        for h in hooks:
            try:
                h(nid, "after", buffer=buffer, inputs=inp, outputs=out)
            except Exception:
                pass

    for step in plan:
        if step[0] == "node":
            _run_node(step[1])
        else:
            _scc_rep, comp = step[1]
            node_order = sorted(comp)
            for h in hooks:
                try:
                    h(_scc_rep, "loop_start", num_steps=n_loop, node_ids=node_order)
                except Exception:
                    pass
            for iteration in range(n_loop):
                for nid in node_order:
                    _run_node(nid)
                for h in hooks:
                    try:
                        h(_scc_rep, "loop_end", iteration=iteration + 1, num_steps=n_loop)
                    except Exception:
                        pass

    # Collect outputs by output spec
    output_spec = graph.get_output_spec()
    result: Dict[str, Any] = {}
    for entry in output_spec:
        key = (entry["node_id"], entry["port_name"])
        out_key = _output_key(entry)
        if key in buffer:
            result[out_key] = buffer[key]
    return result
