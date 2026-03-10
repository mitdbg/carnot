from _internal.chroma_store import ChromaStore
import pdb

embedding_model_name = "Qwen/Qwen3-Embedding-4B"
collection_suffix = "_subset_2"

store = ChromaStore(f"quest_expanded{collection_suffix}", f"./chroma_collections_{embedding_model_name}", embedding_model_name=None)
# store = ChromaStore(f"quest_base{collection_suffix}", f"./chroma_collections_{embedding_model_name}")


def get_doc_by_entity_id(entity_id: str):
    r = store.collection.get(where={"entity_id": entity_id}, include=["metadatas", "documents"], limit=1)
    if not r["ids"]:
        return None
    return {
        "id": r["ids"][0],
        "metadata": r["metadatas"][0],
        "text": r["documents"][0],
    }
    
def get_doc_by_title(title: str):
    r = store.collection.get(where={"title": title}, include=["metadatas", "documents"], limit=1)
    if not r["ids"]:
        return None
    return {
        "id": r["ids"][0],
        "metadata": r["metadatas"][0],
        "text": r["documents"][0],
    }


pdb.set_trace()
doc = get_doc_by_title("White Nights (1985 film)")

doc = get_doc_by_entity_id("Serial_Killing_4_Dummys-046e750d")
print(len(doc["metadata"]))
