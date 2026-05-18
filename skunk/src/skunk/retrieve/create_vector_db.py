import argparse
import json
import os

import chromadb
import numpy as np

CHROMA_MAX_BATCH_SIZE = 5461

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a ChromaDB vector database from precomputed embeddings.")
    parser.add_argument("--embeddings-dir", type=str)
    parser.add_argument("--collection-name", type=str)
    parser.add_argument("--chroma-path", type=str, default=".chromadb",
                        help="Directory to store ChromaDB data (default: .chromadb).")
    args = parser.parse_args()

    # extract args
    embeddings_dir = args.embeddings_dir
    collection_name = args.collection_name
    chroma_path = args.chroma_path

    # initialize chroma client and collection
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(name=collection_name)

    # embeddings directory will contain a list of partition files and metadata dictionaries
    # in one of two possible formats:
    # - embeddings_{idx}.npz and metadata.json
    # - embeddings_{rank_id}_{idx}.npz and metadata_rank{rank_id}.json
    # 
    # first we load the entire metadata dictionary
    metadata = {}
    for file in os.listdir(embeddings_dir):
        if file.startswith("metadata") and file.endswith(".json"):
            with open(os.path.join(embeddings_dir, file)) as f:
                metadata.update(json.load(f))

    # then, we loop through the embedding files and add embeddings to the collection in batches
    for file in os.listdir(embeddings_dir):
        print(f"Processing file {file}...")
        if file.startswith("embeddings") and file.endswith(".npz"):
            # load the embeddings
            embeddings_path = os.path.join(embeddings_dir, file)
            data = np.load(embeddings_path)
            embeddings = data["embeddings"]
            unique_element_ids = data["unique_element_ids"]

            # get the corresponding metadata for these element ids
            file_ids, cleaned_contents, years, months, page_ids, page_keys, elt_ids, elt_types = [], [], [], [], [], [], [], []
            for unique_elt_id in unique_element_ids:
                elt_metadata = metadata[str(unique_elt_id)]
                file_ids.append(elt_metadata["file_id"])
                cleaned_contents.append(elt_metadata["cleaned"])
                years.append(elt_metadata["year"])
                months.append(elt_metadata["month"])
                page_ids.append(elt_metadata["page_id"])
                page_keys.append(elt_metadata["page_key"])
                elt_ids.append(elt_metadata["element_id"])
                elt_types.append(elt_metadata["type"])

            # add to chroma collection in batches
            for i in range(0, len(embeddings), CHROMA_MAX_BATCH_SIZE):
                batch_embeddings = embeddings[i:i+CHROMA_MAX_BATCH_SIZE]
                batch_unique_element_ids = unique_element_ids[i:i+CHROMA_MAX_BATCH_SIZE]
                batch_file_ids = file_ids[i:i+CHROMA_MAX_BATCH_SIZE]
                batch_cleaned_contents = cleaned_contents[i:i+CHROMA_MAX_BATCH_SIZE]
                batch_years = years[i:i+CHROMA_MAX_BATCH_SIZE]
                batch_months = months[i:i+CHROMA_MAX_BATCH_SIZE]
                batch_page_ids = page_ids[i:i+CHROMA_MAX_BATCH_SIZE]
                batch_page_keys = page_keys[i:i+CHROMA_MAX_BATCH_SIZE]
                batch_elt_ids = elt_ids[i:i+CHROMA_MAX_BATCH_SIZE]
                batch_elt_types = elt_types[i:i+CHROMA_MAX_BATCH_SIZE]

                metadata_list = []
                for j in range(len(batch_embeddings)):
                    metadata_list.append({
                        "file_id": batch_file_ids[j],
                        "cleaned": batch_cleaned_contents[j],
                        "year": batch_years[j],
                        "month": batch_months[j],
                        "page_id": batch_page_ids[j],
                        "page_key": batch_page_keys[j],
                        "element_id": batch_elt_ids[j],
                        "type": batch_elt_types[j],
                    })
  
                collection.add(
                    ids=[str(unique_elt_id) for unique_elt_id in batch_unique_element_ids],
                    embeddings=batch_embeddings.tolist(),
                    metadatas=metadata_list,
                )
