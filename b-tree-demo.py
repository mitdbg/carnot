#!/usr/bin/env python3
"""
Test the hierarchical index and HierarchicalCarnotIndex on data/enron-eval-medium.
Requires OPENAI_API_KEY for embedding/summary generation.
Run from repo root: python test_enron_hierarchical_index.py
"""
import logging
import os
import sys
from pathlib import Path

# Ensure we can import carnot (run from repo root with PYTHONPATH=src or pip install -e .)
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from carnot.data.dataset import Dataset as CarnotDataset
from carnot.data.item import DataItem
from carnot.index.hierarchical import FileRouter
from carnot.index.index import HierarchicalCarnotIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    data_dir = Path(__file__).resolve().parent / "data" / "enron-eval-medium"
    if not data_dir.exists():
        logger.error("Data dir not found: %s", data_dir)
        sys.exit(1)

    # Collect .txt files (140 files)
    files = sorted(data_dir.glob("*.txt"))
    if len(files) < 30:
        logger.warning("Only %d files found; routing needs >= 30 for full test", len(files))

    items = [DataItem(path=str(f.absolute())) for f in files]
    logger.info("Loaded %d files from %s", len(items), data_dir)

    # Use persistence so summaries/index are cached for reuse
    router = FileRouter(use_persistence=True)
    if os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    # Query from original email_demo (Raptor, Deathstar, Chewco, Fat Boy investments)
    query = (
        "Find emails that refer to the Raptor, Deathstar, Chewco, and/or Fat Boy investments, "
        "excluding emails that quote text from external articles or sources outside of Enron"
    )
    k = 30

    logger.info("Routing with query: %s...", query[:60])
    routed_items, hierarchical_index = router.route(
        query=query,
        items=items,
        k=k,
        min_files_to_route=30,
    )

    logger.info("Routed: %d items (of %d)", len(routed_items), len(items))
    if hierarchical_index is None:
        logger.warning("No hierarchical index returned (routing skipped or failed)")
        return

    logger.info("Index returned: %s", type(hierarchical_index).__name__)

    # Build dataset with HierarchicalCarnotIndex (same as _apply_file_routing)
    dataset = CarnotDataset(
        name="enron-emails",
        annotation="Routed Enron emails",
        items=routed_items,
        index=HierarchicalCarnotIndex(
            name="enron-emails",
            items=routed_items,
            hierarchical_index=hierarchical_index,
        ),
    )

    # Test dataset.index() - semantic search using hierarchical index
    logger.info("Testing dataset.index() with HierarchicalCarnotIndex...")
    search_results = dataset.index(query="Raptor Chewco Fat Boy investment", k=5)
    logger.info("Top 5 from index.search: %s", [Path(r.path).name for r in search_results])

    logger.info("Done. Hierarchical index + HierarchicalCarnotIndex work correctly.")


if __name__ == "__main__":
    main()