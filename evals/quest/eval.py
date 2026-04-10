import argparse
import json
import os
import random
import time

import carnot


def carnot_run_query(query: dict):
    with open('data/documents.jsonl') as f:
        items = [json.loads(line) for line in f]

    # shuffle items to ensure randomness in selection
    random.shuffle(items)

    # temporary fix to ensure only 100 files are used
    gt_docs = query['docs']
    final_items = [doc for doc in items if doc['title'] in gt_docs]
    while len(final_items) < 100 and len(items) > 0:
        item = items.pop(0)
        if item not in final_items:
            final_items.append(item)

    dataset = carnot.Dataset(
        name="Documents",
        annotation="A set of documents with their titles and content.",
        items=items,
    )
    execution = carnot.Execution(
        query=f"Query: {query['query']}\n\nReturn the (list of) document title(s) under the column `title`.",
        datasets=[dataset],
        llm_config={"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")},
    )
    _, plan = execution.plan()
    execution._plan = plan
    output, _ = execution.run()

    # return output, {"total_tokens": 0, "total_execution_cost": 0.0}
    return [out['title'] for out in output]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate QUEST")
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        help="The domain to evalute (one of 'films' or 'books').",
    )
    parser.add_argument(
        "--queries",
        type=str,
        required=True,
        help="Path to the file containing the queries (one query per line).",
    )
    args = parser.parse_args()

    # extract args
    domain = args.domain
    queries_path = args.queries

    # load queries
    queries = []
    with open(queries_path) as f:
        for line in f:
            d = json.loads(line)
            if d['metadata']['domain'] == domain:
                queries.append(d)

    results = []
    for query in queries:
        # execute the query
        print(f"Executing query: {query['query']}")
        pred_docs = carnot_run_query(query)

        # compute the precision, recall, and F1 score
        gt_docs = query['docs']
        preds = set(pred_docs)
        labels = set(gt_docs)

        tp, fp, fn = 0, 0, 0
        for pred in preds:
            if pred in labels:
                tp += 1
            else:
                fp += 1
        
        for label in labels:
            if label not in preds:
                fn += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        result = {
            "query": query['query'],
            "predicted_docs": pred_docs,
            "ground_truth_docs": gt_docs,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        results.append(result)

        # TODO: remove this break statement after testing
        break

    # save results to a file
    ts = int(time.time())
    with open(f"results_{domain}_{ts}.json", "w") as f:
        json.dump(results, f, indent=4)

    # compute and print average precision, recall, and F1 score
    avg_precision = sum(result['precision'] for result in results) / len(results)
    avg_recall = sum(result['recall'] for result in results) / len(results)
    avg_f1 = sum(result['f1_score'] for result in results) / len(results)
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")
