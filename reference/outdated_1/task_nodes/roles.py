"""
Роли узлов-задач: для auto_connect и role rules.

Канон: WorldGenerator_2.0/TODO_02 A.10, Abstract_Task_Nodes.md, SCALABILITY_AND_EXTENSIBILITY.
- block_type: "backbone", "backbone/unet2d", "solver/ddim", "llm/llama", "conditioner/vlm" и т.д.
- Роль = префикс (backbone, solver, codec, conditioner, tokenizer, adapter, guidance).
- Расписание (noise schedule, sigma/alpha от timestep) — зона ответственности Solver; отдельной роли
  NoiseSchedule нет: один шаг дискретизации и расписание реализуются в узле Solver (конфиг или логика внутри).
- Опциональные роли для расширений: position_embedder, llm, vlm (TODO_07).
"""

from __future__ import annotations

from typing import Set

ROLE_BACKBONE = "backbone"
ROLE_SOLVER = "solver"
ROLE_CODEC = "codec"
ROLE_CONDITIONER = "conditioner"
ROLE_TOKENIZER = "tokenizer"
ROLE_ADAPTER = "adapter"
ROLE_GUIDANCE = "guidance"

# Опциональные роли для расширений (TODO_07): позиционные эмбеддинги, LLM, VLM
ROLE_POSITION_EMBEDDER = "position_embedder"
ROLE_LLM = "llm"
ROLE_VLM = "vlm"

KNOWN_ROLES: Set[str] = {
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
}


def role_from_block_type(block_type: str) -> str | None:
    """
    Derive role from block_type (e.g. "backbone/unet2d" -> "backbone").
    Returns None if block_type does not match any known role prefix.
    """
    if not block_type or not isinstance(block_type, str):
        return None
    bt = block_type.strip().lower()
    for role in KNOWN_ROLES:
        if bt == role or bt.startswith(role + "/") or bt.startswith(role + "_"):
            return role
    return None
