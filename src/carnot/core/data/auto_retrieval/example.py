from __future__ import annotations
import os
import json
import logging
from carnot.core.data.auto_retrieval import SearchClient
from carnot.core.data.auto_retrieval import prepare_quest_queries
from carnot.core.data.auto_retrieval.eval import recall

logging.basicConfig(level=logging.INFO)

documents_path = "dataset/quest/documents.jsonl"
config_path = "config.yaml"
queries = prepare_quest_queries()
print(f"Loaded {len(queries)} queries")

print("Initializing SearchClient...")
client = SearchClient.from_config(config_path)

# print(f"Ingesting documents from {documents_path}...")
# client.ingest_dataset(documents_path)

# print(f"Enriching documents with concepts...")
client.enrich_documents(concepts_path="quest_queries_with_concepts_fewshot_all_clusters100_centroids.json")
import pdb; pdb.set_trace()

for query in queries[:2]:
    results = client.search(query.query, top_k=10, concept_filters=False)
    
    predicted_docs = [
        result.metadata.get("title") if result.metadata and "title" in result.metadata else result.doc_id
        for result in results
    ]
    
    # Get gold document IDs from the QuestQuery object
    gold_docs = query.docs
    
    # Calculate recall
    recall_score = recall(predicted_docs, gold_docs)
    
    print(f"Query: {query.query}")
    print(f"Gold docs: {gold_docs}")
    print(f"Predicted docs: {predicted_docs}")
    print(f"Recall: {recall_score:.4f}")
    # print(f"Results: {results}")
    print()
