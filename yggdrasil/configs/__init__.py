# yggdrasil/configs/__init__.py
"""YAML configs and presets for YggDrasil models.

Usage:
    from yggdrasil.configs import get_recipe, list_recipes, get_preset
    config = get_recipe("sd15_generate")
    model = BlockBuilder.build(config["model"])
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List
from omegaconf import OmegaConf, DictConfig


CONFIG_DIR = Path(__file__).parent
RECIPES_DIR = CONFIG_DIR / "recipes"
BLOCKS_DIR = CONFIG_DIR / "blocks"
USER_DIR = CONFIG_DIR / "user"
# Legacy support
PRESETS_DIR = CONFIG_DIR / "presets"


def list_recipes() -> List[str]:
    """List available recipe configs."""
    if not RECIPES_DIR.exists():
        return []
    return sorted([p.stem for p in RECIPES_DIR.glob("*.yaml")])


def get_recipe(name: str, **overrides) -> DictConfig:
    """Load a recipe config by name.
    
    Args:
        name: Recipe name (without .yaml)
        **overrides: Parameters to override
        
    Returns:
        DictConfig with full pipeline configuration
    """
    path = RECIPES_DIR / f"{name}.yaml"
    if not path.exists():
        available = list_recipes()
        raise FileNotFoundError(
            f"Recipe '{name}' not found. Available: {available}"
        )
    
    config = OmegaConf.load(path)
    if overrides:
        config = OmegaConf.merge(config, overrides)
    return config


def list_presets() -> List[str]:
    """List available presets (legacy, checks both presets/ and recipes/)."""
    presets = []
    for d in [PRESETS_DIR, RECIPES_DIR]:
        if d.exists():
            presets.extend([p.stem for p in d.glob("*.yaml")])
    return sorted(set(presets))


def get_preset(name: str, **overrides) -> DictConfig:
    """Load a preset by name (checks recipes/ then presets/)."""
    for d in [RECIPES_DIR, PRESETS_DIR]:
        path = d / f"{name}.yaml"
        if path.exists():
            config = OmegaConf.load(path)
            if overrides:
                config = OmegaConf.merge(config, overrides)
            return config
    
    available = list_presets()
    raise FileNotFoundError(
        f"Preset '{name}' not found. Available: {available}"
    )


def save_preset(name: str, config: DictConfig | dict, user: bool = True):
    """Save a config preset.
    
    Args:
        name: Preset name
        config: Config to save
        user: If True, save to user/ directory (gitignored)
    """
    target_dir = USER_DIR if user else RECIPES_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{name}.yaml"
    OmegaConf.save(config, path)
    print(f"Preset saved: {path}")
