"""Block Inspector -- CLI tool to inspect registered blocks.

Usage:
    python -m yggdrasil.tools.block_inspector
    python -m yggdrasil.tools.block_inspector --filter backbone
    python -m yggdrasil.tools.block_inspector --detail backbone/dit
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional


def list_blocks(filter_prefix: Optional[str] = None):
    """List all registered blocks."""
    from yggdrasil.core.block.registry import BlockRegistry, auto_discover
    
    auto_discover()
    blocks = BlockRegistry.list_blocks()
    
    if filter_prefix:
        blocks = {k: v for k, v in blocks.items() if k.startswith(filter_prefix)}
    
    # Group by category
    categories = {}
    for key, cls in sorted(blocks.items()):
        cat = key.split("/")[0]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((key, cls))
    
    print(f"\n{'='*60}")
    print(f"  YggDrasil Block Registry ({len(blocks)} blocks)")
    print(f"{'='*60}\n")
    
    for cat, items in sorted(categories.items()):
        print(f"  [{cat}] ({len(items)} blocks)")
        for key, cls in items:
            trainable = "trainable" if getattr(cls, "is_trainable", False) else "frozen"
            print(f"    {key:40s} {cls.__name__:30s} [{trainable}]")
        print()


def inspect_block(block_type: str):
    """Show detailed info about a specific block."""
    from yggdrasil.core.block.registry import BlockRegistry, auto_discover
    
    auto_discover()
    
    try:
        cls = BlockRegistry.get(block_type)
    except KeyError as e:
        print(f"Error: {e}")
        return
    
    print(f"\n{'='*60}")
    print(f"  Block: {block_type}")
    print(f"{'='*60}\n")
    
    print(f"  Class:   {cls.__name__}")
    print(f"  Module:  {cls.__module__}")
    print(f"  Version: {getattr(cls, 'block_version', 'N/A')}")
    
    # Show slots
    try:
        instance = cls.__new__(cls)
        instance.config = {}
        slots = cls._define_slots(instance)
        if slots:
            print(f"\n  Slots:")
            for name, slot in slots.items():
                accepts = getattr(slot.accepts, "__name__", str(slot.accepts))
                optional = "optional" if slot.optional else "required"
                multi = "multiple" if slot.multiple else "single"
                print(f"    {name:20s} accepts={accepts:30s} [{optional}, {multi}]")
                if slot.default:
                    print(f"    {'':20s} default={slot.default}")
        else:
            print(f"\n  Slots: None")
    except Exception:
        print(f"\n  Slots: (could not inspect)")
    
    # Show docstring
    if cls.__doc__:
        print(f"\n  Description:")
        for line in cls.__doc__.strip().split("\n"):
            print(f"    {line.strip()}")
    
    # Show MRO
    print(f"\n  Inheritance chain:")
    for c in cls.__mro__[:-1]:
        print(f"    {c.__name__}")
    print()


def list_plugins():
    """List all registered plugins."""
    try:
        from yggdrasil.plugins.base import PluginRegistry
        import yggdrasil.plugins  # Trigger auto-discovery
    except ImportError:
        print("Plugin system not available")
        return
    
    plugins = PluginRegistry.list_plugins()
    print(f"\n{'='*60}")
    print(f"  YggDrasil Plugins ({len(plugins)} plugins)")
    print(f"{'='*60}\n")
    
    for name, cls in sorted(plugins.items()):
        desc = getattr(cls, "description", "")
        ver = getattr(cls, "version", "1.0.0")
        print(f"  {name:15s} v{ver:8s} {desc}")
    print()


def main():
    parser = argparse.ArgumentParser(description="YggDrasil Block Inspector")
    parser.add_argument("--filter", "-f", help="Filter blocks by prefix (e.g. 'backbone')")
    parser.add_argument("--detail", "-d", help="Show detailed info for a specific block type")
    parser.add_argument("--plugins", "-p", action="store_true", help="List plugins")
    
    args = parser.parse_args()
    
    if args.plugins:
        list_plugins()
    elif args.detail:
        inspect_block(args.detail)
    else:
        list_blocks(args.filter)


if __name__ == "__main__":
    main()
