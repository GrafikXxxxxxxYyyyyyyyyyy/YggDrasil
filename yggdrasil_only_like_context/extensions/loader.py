# yggdrasil/extensions/loader.py
"""Extension loader — auto-discovers and loads custom block packages.

Extension directories scanned (in order):
1. ./extensions/            (project-local)
2. ~/.yggdrasil/extensions/ (user-global)
3. Custom paths via load_extensions(paths=[...])

Each extension is a Python package with:
- __init__.py containing @register_block decorated classes
- Optional manifest.yaml with metadata

Example extension structure::

    extensions/
      my_upscaler/
        __init__.py          # @register_block("my/upscaler") class ...
        manifest.yaml        # name, version, blocks, requires
        model.py             # Additional code

Example manifest.yaml::

    name: my_upscaler
    version: 1.0.0
    author: Your Name
    description: Super resolution upscaler
    blocks:
      - my/upscaler
      - my/face_restore
    requires:
      - realesrgan>=0.3.0
"""
from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default search directories
_DEFAULT_DIRS = [
    Path("./extensions"),
    Path.home() / ".yggdrasil" / "extensions",
]

# Loaded extensions cache
_loaded_extensions: List["ExtensionInfo"] = []


@dataclass
class ExtensionInfo:
    """Metadata about a loaded extension."""
    name: str
    version: str = "0.0.0"
    author: str = ""
    description: str = ""
    path: str = ""
    blocks: List[str] = field(default_factory=list)
    requires: List[str] = field(default_factory=list)
    loaded: bool = False
    error: Optional[str] = None


def load_extensions(
    paths: Optional[List[str | Path]] = None,
    skip_errors: bool = True,
) -> List[ExtensionInfo]:
    """Load all extensions from directories.
    
    Args:
        paths: Additional directories to scan (on top of defaults)
        skip_errors: If True, log errors but continue loading (default True)
    
    Returns:
        List of ExtensionInfo for all discovered extensions.
    """
    global _loaded_extensions
    
    search_dirs = list(_DEFAULT_DIRS)
    if paths:
        search_dirs.extend([Path(p) for p in paths])
    
    extensions = []
    
    for ext_dir in search_dirs:
        ext_dir = Path(ext_dir).resolve()
        if not ext_dir.is_dir():
            continue
        
        logger.debug(f"Scanning extensions in: {ext_dir}")
        
        for item in sorted(ext_dir.iterdir()):
            if not item.is_dir():
                continue
            if item.name.startswith((".", "_", "__")):
                continue
            
            init_file = item / "__init__.py"
            if not init_file.exists():
                continue
            
            ext_info = _load_single_extension(item, skip_errors)
            extensions.append(ext_info)
    
    _loaded_extensions = extensions
    
    loaded_count = sum(1 for e in extensions if e.loaded)
    failed_count = sum(1 for e in extensions if not e.loaded)
    
    if loaded_count > 0:
        logger.info(f"Loaded {loaded_count} extensions")
    if failed_count > 0:
        logger.warning(f"Failed to load {failed_count} extensions")
    
    return extensions


def _load_single_extension(ext_path: Path, skip_errors: bool) -> ExtensionInfo:
    """Load a single extension package."""
    # Read manifest if available
    info = _read_manifest(ext_path)
    info.path = str(ext_path)
    
    try:
        # Check requirements
        _check_requirements(info.requires)
        
        # Add to sys.path if not already there
        parent = str(ext_path.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        
        # Import the package
        module_name = ext_path.name
        
        # Use importlib to load
        spec = importlib.util.spec_from_file_location(
            f"yggdrasil_ext_{module_name}",
            ext_path / "__init__.py",
            submodule_search_locations=[str(ext_path)],
        )
        
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec for {ext_path}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        
        # Discover registered blocks from the module
        from yggdrasil.core.block.registry import BlockRegistry
        
        if not info.blocks:
            # Auto-detect blocks by checking what was registered
            for block_type, block_cls in BlockRegistry._registry.items():
                mod = getattr(block_cls, "__module__", "")
                if module_name in mod or spec.name in mod:
                    info.blocks.append(block_type)
        
        info.loaded = True
        logger.info(
            f"Loaded extension '{info.name}' v{info.version} "
            f"({len(info.blocks)} blocks: {info.blocks})"
        )
        
    except Exception as e:
        info.loaded = False
        info.error = f"{type(e).__name__}: {e}"
        msg = f"Failed to load extension '{info.name}' from {ext_path}: {info.error}"
        if skip_errors:
            logger.warning(msg)
        else:
            raise ImportError(msg) from e
    
    return info


def _read_manifest(ext_path: Path) -> ExtensionInfo:
    """Read extension manifest if it exists."""
    manifest_path = ext_path / "manifest.yaml"
    
    info = ExtensionInfo(name=ext_path.name)
    
    if manifest_path.exists():
        try:
            from omegaconf import OmegaConf
            manifest = OmegaConf.to_container(OmegaConf.load(manifest_path), resolve=True)
            info.name = manifest.get("name", ext_path.name)
            info.version = str(manifest.get("version", "0.0.0"))
            info.author = manifest.get("author", "")
            info.description = manifest.get("description", "")
            info.blocks = list(manifest.get("blocks", []))
            info.requires = list(manifest.get("requires", []))
        except Exception as e:
            logger.debug(f"Could not read manifest for {ext_path.name}: {e}")
    
    return info


def _check_requirements(requires: List[str]):
    """Check that required packages are installed."""
    if not requires:
        return
    
    try:
        import pkg_resources
        for req in requires:
            try:
                pkg_resources.require(req)
            except pkg_resources.DistributionNotFound:
                raise ImportError(
                    f"Missing required package: {req}. "
                    f"Install with: pip install {req}"
                )
            except pkg_resources.VersionConflict as e:
                logger.warning(f"Version conflict: {e}")
    except ImportError:
        pass  # pkg_resources not available


def list_extensions() -> List[ExtensionInfo]:
    """List all loaded extensions."""
    if not _loaded_extensions:
        load_extensions()
    return _loaded_extensions


def get_extension(name: str) -> Optional[ExtensionInfo]:
    """Get extension info by name."""
    for ext in list_extensions():
        if ext.name == name:
            return ext
    return None
