import os
import sys
import yaml
import logging
import argparse

# SQLite compatibility
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import from lib
from lib.decomposition_generator import generate_strategies
from lib.decomposition_executor import execute_all_strategies
from lib.analyze_decomposition_results import analyze_results

CONFIG_PATH = "config.yaml"

def load_config(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Run Decomposition Pipeline")
    parser.add_argument(
        "--mode", 
        choices=["generate", "execute", "analyze", "auto"], 
        default="generate",
        help=(
            "generate: Create strategy files only (then stop for review). "
            "execute: Run existing strategy files and analyze results. "
            "analyze: Run analysis on existing predictions only. "
            "auto: Run the full pipeline without stopping (use with caution)."
        )
    )
    args = parser.parse_args()

    logger.info(f"Loading configuration from {CONFIG_PATH}...")
    config = load_config(CONFIG_PATH)

    # --- Paths Setup ---
    queries_path = config['data']['queries_file']
    subset_name = os.path.splitext(os.path.basename(queries_path))[0]
    target_k = config['decomposition'].get('top_k', 100)
    
    strategies_dir = f"decomposition_strategies/{subset_name}_k{target_k}"
    pred_output_path = f"decomposition/pred_set_ops_{subset_name}.jsonl"
    report_output_path = f"decomposition/results/recall_report_{subset_name}.txt"

    # ==========================================
    # MODE: GENERATE (Default)
    # ==========================================
    if args.mode in ["generate", "auto"]:
        logger.info("=== Step 1: Generating Decompositions ===")
        generate_strategies(
            queries_path=queries_path,
            output_dir=strategies_dir,
            target_k=target_k,
            llm_model=config.get('decomposition', {}).get('llm_model', 'gpt-4o-mini') 
        ) #
        
        if args.mode == "generate":
            logger.info("\n" + "="*60)
            logger.info(f"GENERATION COMPLETE. Strategies saved to: {strategies_dir}")
            logger.info("IMPORTANT: Please review/edit the generated Python files.")
            logger.info("When ready, run this script again with: --mode execute")
            logger.info("="*60 + "\n")
            return

    # ==========================================
    # MODE: EXECUTE
    # ==========================================
    if args.mode in ["execute", "auto"]:
        logger.info("=== Step 2: Executing Strategies ===")
        
        if not os.path.exists(strategies_dir) or not os.listdir(strategies_dir):
            logger.error(f"No strategies found in {strategies_dir}. Run with --mode generate first.")
            return

        execute_all_strategies(
            strategies_dir=strategies_dir,
            queries_path=queries_path,
            output_path=pred_output_path,
            config=config
        ) #

    # ==========================================
    # MODE: ANALYZE (Runs automatically after execute)
    # ==========================================
    if args.mode in ["execute", "analyze", "auto"]:
        logger.info("=== Step 3: Analyzing Results ===")
        if os.path.exists(pred_output_path):
            analyze_results(
                gold_path=queries_path,
                pred_path=pred_output_path,
                output_report_path=report_output_path
            ) #
        else:
            logger.error(f"Prediction file not found at {pred_output_path}. Cannot analyze.")

    logger.info("Decomposition pipeline tasks completed.")

if __name__ == "__main__":
    main()