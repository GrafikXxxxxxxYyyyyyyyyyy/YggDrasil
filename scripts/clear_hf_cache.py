#!/usr/bin/env python3
"""Очистка кэша Hugging Face.

По умолчанию удаляет весь кэш (каталог ~/.cache/huggingface/hub или HF_HUB_CACHE).
Опционально можно удалить только указанные репозитории.

Использование:
  python scripts/clear_hf_cache.py              # очистить весь кэш
  python scripts/clear_hf_cache.py --dry-run   # показать размер, не удалять
  python scripts/clear_hf_cache.py black-forest-labs/FLUX.2-klein-9B   # только эту модель
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Clear Hugging Face cache")
    parser.add_argument(
        "repos",
        nargs="*",
        help="Repo IDs to remove (e.g. black-forest-labs/FLUX.2-klein-9B). If empty, clear all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be deleted, do not delete.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Override cache directory (default: HF_HUB_CACHE or ~/.cache/huggingface/hub).",
    )
    parser.add_argument("-y", "--yes", action="store_true", help="Do not ask for confirmation.")
    args = parser.parse_args()

    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        print("Install: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    cache_dir = args.cache_dir
    info = scan_cache_dir(cache_dir=cache_dir)

    if not info.repos:
        print("Cache is empty.")
        return 0

    # Collect revisions to delete: all or only matching repos
    revs_to_delete = []
    repo_ids_to_match = [r.strip().lower() for r in args.repos] if args.repos else None

    for repo in info.repos:
        if repo_ids_to_match is None:
            revs_to_delete.extend(r.commit_hash for r in repo.revisions)
            print(f"  {repo.repo_id}: {repo.size_on_disk_str}")
        elif repo.repo_id.lower() in repo_ids_to_match or any(
            repo.repo_id.lower().endswith(r) or r in repo.repo_id.lower()
            for r in repo_ids_to_match
        ):
            revs_to_delete.extend(r.commit_hash for r in repo.revisions)
            print(f"  {repo.repo_id}: {repo.size_on_disk_str}")

    if not revs_to_delete:
        if repo_ids_to_match:
            print("No matching repos in cache.")
        return 0

    strategy = info.delete_revisions(*revs_to_delete)
    print(f"Would free: {strategy.expected_freed_size_str}")

    if args.dry_run:
        print("Dry-run: nothing deleted.")
        return 0

    if not args.yes:
        confirm = input("Delete cache? [y/N] ").strip().lower()
        if confirm != "y" and confirm != "yes":
            print("Cancelled.")
            return 0

    strategy.execute()
    print("Cache cleared.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
