"""
Task nodes (Abstract Task Nodes): Block + role for graph assembly.

Canon: WorldGenerator_2.0/Abstract_Task_Nodes.md, TODO_02.
- Abstract Backbone, Solver, Codec, Conditioner, Tokenizer, Adapter, Guidance
- Role rules for auto_connect; registration under block_type (e.g. backbone/unet2d).
"""

# Register identity stubs and template builders when package is used
from yggdrasill.task_nodes import stubs  # noqa: F401 - registers stub block types
from yggdrasill.task_nodes import templates  # noqa: F401 - registers Graph.from_template names

from yggdrasill.task_nodes.roles import (
    ROLE_BACKBONE,
    ROLE_SOLVER,
    ROLE_CODEC,
    ROLE_CONDITIONER,
    ROLE_TOKENIZER,
    ROLE_ADAPTER,
    ROLE_GUIDANCE,
    ROLE_POSITION_EMBEDDER,
    ROLE_LLM,
    ROLE_VLM,
    role_from_block_type,
    KNOWN_ROLES,
)
from yggdrasill.task_nodes.role_rules import (
    get_rule_edges,
    suggest_edges_for_new_node,
)
from yggdrasill.task_nodes.abstract import (
    AbstractBackbone,
    AbstractSolver,
    AbstractCodec,
    AbstractConditioner,
    AbstractTokenizer,
    AbstractAdapter,
    AbstractGuidance,
)
from yggdrasill.task_nodes.auto_connect import apply_auto_connect, use_task_node_auto_connect

__all__ = [
    "ROLE_BACKBONE",
    "ROLE_SOLVER",
    "ROLE_CODEC",
    "ROLE_CONDITIONER",
    "ROLE_TOKENIZER",
    "ROLE_ADAPTER",
    "ROLE_GUIDANCE",
    "ROLE_POSITION_EMBEDDER",
    "ROLE_LLM",
    "ROLE_VLM",
    "KNOWN_ROLES",
    "role_from_block_type",
    "get_rule_edges",
    "suggest_edges_for_new_node",
    "AbstractBackbone",
    "AbstractSolver",
    "AbstractCodec",
    "AbstractConditioner",
    "AbstractTokenizer",
    "AbstractAdapter",
    "AbstractGuidance",
    "apply_auto_connect",
    "use_task_node_auto_connect",
]
