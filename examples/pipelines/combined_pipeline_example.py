"""
Пример многостадийного (комбинированного) пайплайна. REFACTORING_GRAPH_PIPELINE_ENGINE.md §11.7 P1–P4, Фаза 9.

InferencePipeline принимает один граф, список графов или словарь графов.
Отдельного класса CombinedPipeline нет — комбинированный сценарий = тот же InferencePipeline.

Варианты:
  - graphs=[g1, g2] — этапы по порядку; выход g1 → вход g2 по connections или default_link.
  - graphs={"seg": g1, "gen": g2} — именованные этапы; parallel_groups задаёт план выполнения.
"""
from __future__ import annotations


def build_combined_from_list():
    """Два графа по порядку: pipe = InferencePipeline(graphs=[(name, g), ...], connections=...)."""
    from yggdrasil.pipeline import InferencePipeline
    from yggdrasil.core.graph.graph import ComputeGraph

    # В реальном сценарии: g1 = ComputeGraph.from_template("segmenter"), g2 = ComputeGraph.from_template("sdxl_txt2img")
    g1 = ComputeGraph("stage1")
    g2 = ComputeGraph("stage2")
    # graphs= список пар (имя_этапа, граф); при пустых connections — цепочка stage0.output -> stage1.input
    pipe = InferencePipeline(
        graphs=[("stage1", g1), ("stage2", g2)],
        connections=[],  # или явные рёбра (src, src_port, dst, dst_port)
        inputs={},
        outputs={},
    )
    return pipe


def build_combined_from_dict_with_parallel_groups():
    """Именованные этапы и parallel_groups: порядок выполнения по уровням (§11.7 P3)."""
    from yggdrasil.pipeline import InferencePipeline
    from yggdrasil.core.graph.graph import ComputeGraph

    g_seg = ComputeGraph("segmenter")
    g_gen = ComputeGraph("generator")
    # parallel_groups=[["segmenter"], ["generator"]] — сначала segmenter, потом generator.
    # Для параллельных веток: parallel_groups=[["branch_a", "branch_b"]] — оба в одном уровне.
    pipe = InferencePipeline(
        graphs={"segmenter": g_seg, "generator": g_gen},
        connections=[],  # при пустых — default_link между этапами по порядку (если реализован)
        inputs={},
        outputs={},
        parallel_groups=[["segmenter"], ["generator"]],
    )
    return pipe


def build_from_spec_combined():
    """Единая точка входа: InferencePipeline.from_spec(spec) при spec = список (name, graph) или словарь."""
    from yggdrasil.pipeline import InferencePipeline
    from yggdrasil.core.graph.graph import ComputeGraph

    g1 = ComputeGraph("a")
    g2 = ComputeGraph("b")
    # from_spec(list) вызывает from_combined(stages=...); stages = [(name, graph), ...]
    pipe = InferencePipeline.from_spec([("stage_a", g1), ("stage_b", g2)], inputs={}, outputs={})
    return pipe


if __name__ == "__main__":
    try:
        import torch  # noqa: F401
    except ImportError:
        print("Skip run: torch not installed. Example API is valid; install torch and yggdrasil to run.")
        exit(0)
    # Демо: создание комбинированного пайплайна (без запуска инференса)
    p1 = build_combined_from_list()
    print("InferencePipeline(graphs=[(name, g), ...]):", p1.graph.name, "nodes:", list(p1.graph.nodes.keys()))

    p2 = build_combined_from_dict_with_parallel_groups()
    print("InferencePipeline(graphs={...}, parallel_groups=...):", p2.graph.metadata.get("parallel_groups"))

    p3 = build_from_spec_combined()
    print("from_spec([('stage_a', g1), ('stage_b', g2)]):", p3.graph.name)
