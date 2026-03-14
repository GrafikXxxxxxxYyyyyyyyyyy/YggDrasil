# Yggdrasill

Hypergraph framework from block to universe.

A domain-agnostic engine for building, validating, planning, and executing
computational graphs — from single task-nodes through hypergraphs to
multi-level workflows. Designed for diffusion models, LLMs, agent loops,
and any directed computation pipeline.

## Installation

```bash
pip install -e .

# With development dependencies:
pip install -e ".[dev]"

# With YAML config support:
pip install -e ".[yaml]"
```

## Running tests

```bash
pytest tests/
```

## Quick start

### Build a minimal hypergraph

```python
from yggdrasill.engine.structure import Hypergraph
from yggdrasill.engine.edge import Edge
import yggdrasill.task_nodes  # registers identity stubs

h = Hypergraph(graph_id="demo")
h.add_node_from_config("enc", "converter/identity")
h.add_node_from_config("bb", "backbone/identity")
h.add_edge(Edge("enc", "output", "bb", "latent"))
h.expose_input("enc", "input", "x")
h.expose_input("bb", "timestep", "t")
h.expose_output("bb", "pred", "y")

result = h.run({"x": "data", "t": 0})
print(result)  # {"y": "data"}
```

### Build a workflow (hypergraph of hypergraphs)

```python
from yggdrasill.workflow import Workflow

w = Workflow(workflow_id="pipeline")
w.add_node("step1", h1)  # h1 is a Hypergraph
w.add_node("step2", h2)
w.add_edge("step1", "y", "step2", "x")
w.infer_exposed_ports()

result = w.run({"x": "input"})
```

### Save and load

```python
h.save("/tmp/my_graph")
h2 = Hypergraph.load("/tmp/my_graph")
assert h2.run({"x": "data", "t": 0}) == result
```

## Architecture

- **Foundation**: `AbstractBaseBlock` (data/computation) + `AbstractGraphNode` (graph position/ports) — dual inheritance, no wrappers
- **Engine**: Universal `Validator` → `Planner` → `Executor` pipeline operating on any structural protocol
- **Task Nodes**: Seven abstract roles (Backbone, Injector, Conjector, InnerModule, OuterModule, Helper, Converter) with canonical port contracts
- **Serialization**: Config (JSON/YAML) + Checkpoint (pickle) with block_id deduplication
- **Workflow**: Hypergraph-of-hypergraphs executed by the same engine

## License

Apache-2.0
