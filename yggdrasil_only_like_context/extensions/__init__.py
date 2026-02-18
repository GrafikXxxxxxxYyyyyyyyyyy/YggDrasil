"""YggDrasil Extension System — drop-in custom blocks without forking.

Like ComfyUI's custom_nodes: drop a Python package into extensions/ and
it's automatically loaded on startup.

Usage::

    from yggdrasil.extensions import load_extensions, list_extensions
    
    # Load all extensions from default directories
    loaded = load_extensions()
    
    # List installed extensions
    for ext in list_extensions():
        print(f"{ext.name} v{ext.version}: {ext.blocks}")
"""
from .loader import load_extensions, list_extensions, ExtensionInfo

__all__ = ["load_extensions", "list_extensions", "ExtensionInfo"]
