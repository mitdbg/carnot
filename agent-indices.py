import logging
import os
from pathlib import Path

import carnot
from carnot.data.dataset import Dataset
from carnot.data.item import DataItem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 1. Load text files (Enron or any .txt)
data_dir = Path("data/enron-eval-medium")  # or your text file directory
items = [DataItem(path=str(f.absolute())) for f in sorted(data_dir.glob("*.txt"))]

logger.info("Items loaded: %d", len(items))
# 3. Create dataset
dataset = Dataset(name="emails", annotation="Email documents", items=items)
logger.info("Dataset created")
# 4. Create Execution WITH indices (this adds IndexSearchTool)
exec_instance = carnot.Execution(
    query="Give me files related to Raptor or Deathstar investments",
    datasets=[dataset],
    llm_config={"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")},
)
logger.info("Execution created")
# 5. Plan and run
nl_plan, plan = exec_instance.plan()
exec_instance._plan = plan
items_out, answer_str = exec_instance.run()
logger.info("Answer: %s", answer_str)