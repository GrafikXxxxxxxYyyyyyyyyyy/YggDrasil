#!/usr/bin/env python3
"""Список зарегистрированных шаблонов графов и типов блоков.

Помогает узнать, какие пайплайны и блоки доступны для сборки и замены.
Шаблоны используются в ComputeGraph.from_template(name).
Типы блоков — в BlockBuilder.build({"type": "..."}).

Запуск:
    python examples/images/sd15/10_list_templates_blocks.py
"""
from yggdrasil.core.graph.graph import ComputeGraph
from yggdrasil.core.graph.templates import list_templates
from yggdrasil.core.block.registry import list_blocks


def main():
    print("=" * 60)
    print("ШАБЛОНЫ ГРАФОВ (ComputeGraph.from_template)")
    print("=" * 60)
    all_templates = list_templates()
    for name in all_templates:
        if "sd15" in name:
            print(f"  {name}")
    print("  ... и другие (sdxl, flux, controlnet, etc.)")
    print(f"\nВсего шаблонов: {len(all_templates)}")
    for t in all_templates[:30]:
        print(f"  {t}")
    if len(all_templates) > 30:
        print(f"  ... и ещё {len(all_templates) - 30}")

    print("\n" + "=" * 60)
    print("ТИПЫ БЛОКОВ (BlockBuilder.build({\"type\": \"...\"}))")
    print("=" * 60)
    blocks = list_blocks()
    sd_relevant = [k for k in blocks if any(x in k for x in ["backbone", "guidance", "conditioner", "codec", "diffusion/solver", "adapter"])]
    for b in sorted(sd_relevant):
        print(f"  {b}")
    print(f"\nВсего типов блоков: {len(blocks)}")


if __name__ == "__main__":
    main()
