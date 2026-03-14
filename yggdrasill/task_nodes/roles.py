from __future__ import annotations

from enum import Enum
from typing import Optional


class Role(Enum):
    BACKBONE = "backbone"
    INJECTOR = "injector"
    CONJECTOR = "conjector"
    INNER_MODULE = "inner_module"
    OUTER_MODULE = "outer_module"
    HELPER = "helper"
    CONVERTER = "converter"


_PREFIX_TO_ROLE = {r.value: r for r in Role}


def role_from_block_type(block_type: str) -> Optional[Role]:
    """Extract the role from a block_type string like 'backbone/identity'."""
    prefix = block_type.split("/", 1)[0].lower()
    return _PREFIX_TO_ROLE.get(prefix)


ALL_ROLES = list(Role)
