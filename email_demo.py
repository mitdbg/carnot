import os
import time

import chromadb

import carnot


def clear_cache():
    # remove context files
    context_dir = os.path.join(carnot.constants.PZ_DIR, "contexts")
    for filename in os.listdir(context_dir):
        os.remove(os.path.join(context_dir, filename))

    # clear collection
    chroma_dir = os.path.join(carnot.constants.PZ_DIR, "chroma")
    chroma_client = chromadb.PersistentClient(chroma_dir)
    try:  # noqa: SIM105
        chroma_client.delete_collection("contexts")
    except chromadb.errors.NotFoundError:
        # collection does not exist, no need to delete
        pass

def run_carnot():
    ds = carnot.TextFileContext("data/enron-eval-medium", "enron-data", "250 emails from Enron employees")
    ds = ds.search(
        "Find emails that refer to the Raptor, Deathstar, Chewco, and/or Fat Boy investments, "
        "excluding emails that quote text from external articles or sources outside of Enron"
    )
    config = carnot.QueryProcessorConfig(
        policy=carnot.MaxQuality(),
        progress=False,
    )
    return ds.run(config=config)


if __name__ == "__main__":
    clear_cache()
    print("Cache cleared")

    start_time = time.time()

    # execute script
    output = run_carnot()
    output.to_df().to_csv("carnot-email-output.csv", index=False)

    print(f"TOTAL TIME: {time.time() - start_time:.2f}")
