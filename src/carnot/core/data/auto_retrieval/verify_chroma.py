import chromadb
import os
import sys
import argparse

def verify_chroma(remove_concepts: bool = False):
    persist_dir = "./chroma_quest"
    collection_name = "quest_documents"
    
    if not os.path.exists(persist_dir):
        print(f"Error: Persist directory '{persist_dir}' does not exist.")
        return

    try:
        client = chromadb.PersistentClient(path=persist_dir)
        try:
            collection = client.get_collection(collection_name)
        except Exception as e:
            print(f"Error: Collection '{collection_name}' not found. {e}")
            return

        count = collection.count()
        print(f"Collection '{collection_name}' has {count} documents.")

        if remove_concepts:
            print("\nRemoving 'concept:*' metadata from all documents...")
            # We must fetch ALL IDs to ensure we clean everything.
            all_ids = collection.get(limit=count)['ids']
            if not all_ids:
                print("No documents found.")
                return
            
            print(f"Fetched {len(all_ids)} document IDs. Starting batch cleanup...")

            # Batch process for safety
            batch_size = 256 # Smaller batch size because we are fetching embeddings
            total_removed = 0
            
            for i in range(0, len(all_ids), batch_size):
                batch_ids = all_ids[i:i+batch_size]
                
                # Fetch FULL data because we need to Delete+Add to remove metadata keys
                batch_data = collection.get(ids=batch_ids, include=['metadatas', 'documents', 'embeddings'])
                
                metadatas = batch_data['metadatas']
                documents = batch_data['documents']
                embeddings = batch_data['embeddings']
                
                updated_metadatas = []
                ids_to_update = []
                docs_to_update = []
                embs_to_update = []
                
                needs_update = False
                
                for idx, meta in enumerate(metadatas):
                    doc_id = batch_ids[idx]
                    if not meta:
                        updated_metadatas.append({})
                        ids_to_update.append(doc_id)
                        docs_to_update.append(documents[idx])
                        embs_to_update.append(embeddings[idx])
                        continue
                        
                    # Filter out keys starting with "concept:"
                    new_meta = {k: v for k, v in meta.items() if not k.startswith("concept:")}
                    
                    updated_metadatas.append(new_meta)
                    ids_to_update.append(doc_id)
                    docs_to_update.append(documents[idx])
                    embs_to_update.append(embeddings[idx])
                    
                    if len(new_meta) != len(meta):
                        needs_update = True
                
                if needs_update:
                    # ChromaDB update() merges metadata, it doesn't replace.
                    # To delete keys, we must DELETE and ADD back.
                    collection.delete(ids=ids_to_update)
                    collection.add(
                        ids=ids_to_update,
                        embeddings=embs_to_update,
                        metadatas=updated_metadatas,
                        documents=docs_to_update
                    )
                    total_removed += len(ids_to_update)
                
                print(f"Processed {min(i + batch_size, len(all_ids))}/{len(all_ids)} documents...")
                
            print(f"\nSuccessfully cleaned metadata for {total_removed} documents (via delete+add).")

        if count > 0:
            limit_peek = 5
            result = collection.peek(limit=limit_peek)
            print(f"\nSample Document Keys (showing first of {limit_peek} samples):")
            print(result.keys())
            
            num_returned = len(result['ids'])
            for i in range(num_returned):
                print(f"\n--- Document {i+1} ---")
                if 'ids' in result and result['ids']:
                    print(f"ID: {result['ids'][i]}")
                if 'documents' in result and result['documents']:
                    # Truncate for display
                    doc_preview = result['documents'][i][:50] + "..." if len(result['documents'][i]) > 50 else result['documents'][i]
                    print(f"Document (snippet): {doc_preview}")
                if 'embeddings' in result and len(result['embeddings']) > i:
                    print(f"Embedding length: {len(result['embeddings'][i])}")
                if 'metadatas' in result and result['metadatas']:
                    print(f"Metadata: {result['metadatas'][i]}")
        else:
            print("Collection is empty.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify and manage ChromaDB collection.")
    parser.add_argument("--remove-concepts", action="store_true", help="Remove all 'concept:*' metadata fields.")
    args = parser.parse_args()
    
    verify_chroma(remove_concepts=args.remove_concepts)

