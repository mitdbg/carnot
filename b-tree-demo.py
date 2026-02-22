#!/usr/bin/env python3
"""Test hierarchical index and routing on enron-eval-medium."""
import logging
import os
import sys
from pathlib import Path

import carnot

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from carnot.data.dataset import Dataset as CarnotDataset
from carnot.data.item import DataItem
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

    files = sorted(data_dir.glob("*.txt"))
    if len(files) < 30:
        logger.warning("Only %d files found; routing needs >= 30 for full test", len(files))

    items = [DataItem(path=str(f.absolute())) for f in files]
    logger.info("Loaded %d files from %s", len(items), data_dir)

    if os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    query = (
        "Find emails that refer to the Raptor, Deathstar, Chewco, and/or Fat Boy investments, "
        "excluding emails that quote text from external articles or sources outside of Enron"
    )

    carnot_index = HierarchicalCarnotIndex(
        name="enron-emails",
        items=items,
        use_persistence=True,
    )
    if len(items) >= 30:
        logger.info("Routing with query: %s...", query[:60])
        routed_items = carnot_index.search(query, k=50)
    else:
        logger.info("Skipping routing (< 30 files), using all items")
        routed_items = list(items)

    logger.info("Routed: %d items (of %d)", len(routed_items), len(items))
    logger.info("Routed file paths: %s", [r.path for r in routed_items])

    dataset = CarnotDataset(
        name="enron-emails",
        annotation="Routed Enron emails",
        items=routed_items,
        indices={
            "hierarchical": HierarchicalCarnotIndex(
                name="enron-emails",
                items=routed_items,
                hierarchical_index=carnot_index._hierarchical,
            ),
        },
    )
    logger.info("Generating plan and executing...")
    exec_instance = carnot.Execution(
        query=query,
        datasets=[dataset],
        llm_config={"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")},
    )
    _, plan = exec_instance.plan()
    exec_instance._plan = plan
    items, answer_str = exec_instance.run()

    logger.info("Answer:\n%s", answer_str)


if __name__ == "__main__":
    main()