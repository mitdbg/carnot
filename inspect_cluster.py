#!/usr/bin/env python3
"""List leaf clusters and their summaries from the persisted hierarchical index."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from carnot.index.hierarchical_types import InternalNode
from carnot.index.persistence import HierarchicalIndexCache


def collect_leaf_clusters(root_level, max_depth=2, current_depth=0):
    """Collect nodes with is_leaf_cluster=True within max_depth levels."""
    clusters = []
    for node in root_level:
        if hasattr(node, "path"):
            continue
        if node.is_leaf_cluster:
            clusters.append((current_depth, node))
        elif node.children and current_depth < max_depth - 1:
            clusters.extend(
                collect_leaf_clusters(node.children, max_depth, current_depth + 1)
            )
    return clusters


def main():
    data_dir = Path(__file__).resolve().parent / "data" / "enron-eval-medium"
    if not data_dir.exists():
        print("Data dir not found:", data_dir)
        sys.exit(1)

    paths = [str(f.absolute()) for f in sorted(data_dir.glob("*.txt"))]
    cache = HierarchicalIndexCache()
    index = cache.load(paths)
    if index is None:
        print("No cached index for this path set. Run b-tree-demo.py first.")
        sys.exit(1)

    clusters = collect_leaf_clusters(index._root_level, max_depth=2)
    for i, (depth, node) in enumerate(clusters):
        print(f"\n--- Cluster {i + 1} (depth {depth}) ---")
        print(f"Summary: {node.summary[:2200]}..." if len(node.summary) > 200 else f"Summary: {node.summary}")
        print(f"Files ({len(node.child_paths)}):")
        for p in node.child_paths:
            print(f"  {Path(p).name}")
        print()


if __name__ == "__main__":
    main()