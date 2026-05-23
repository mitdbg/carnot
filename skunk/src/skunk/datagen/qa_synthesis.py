import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the question-answer synthesis script.")
    parser.add_argument(
        "--input-csv",
        type=str,
        default="officeqa_pro.csv",
        help="Path to the input CSV file containing questions and answers (default: officeqa_pro.csv)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="qa_synthesis_output.json",
        help="Path to the output JSON file (default: qa_synthesis_output.json)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="google/gemini-3.5-flash",
        help="ID of the language model to use for synthesis (default: google/gemini-3.5-flash)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of synthesized QA pairs to generate.",
    )
    args = parser.parse_args()

    # load the OfficeQA Pro questions from the CSV file.
    officeqa_df = pd.read_csv(args.input_csv)

    # synthesize num_samples new QA pairs using the specified language model
    