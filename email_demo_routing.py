import os
import time
import logging
import chromadb
import carnot

# Configure logging to see routing info
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def clear_cache():
    """Clear context cache before running (same as original email_demo.py)"""
    # remove context files
    context_dir = os.path.join(carnot.constants.PZ_DIR, "contexts")
    if os.path.exists(context_dir):
        for filename in os.listdir(context_dir):
            os.remove(os.path.join(context_dir, filename))

    # clear collection
    chroma_dir = os.path.join(carnot.constants.PZ_DIR, "chroma")
    chroma_client = chromadb.PersistentClient(chroma_dir)
    try:
        chroma_client.delete_collection("contexts")
    except chromadb.errors.InvalidCollectionException:
        pass

print("=" * 80)
print("TESTING FLAT FILE INDEX ROUTING")
print("=" * 80)

# Clear cache first
print("\n0. Clearing context cache...")
clear_cache()

start_time = time.time()

# Create context with routing enabled (default)
print("\n1. Creating TextFileContext with routing enabled...")
ds = carnot.TextFileContext(
    "data/enron-eval-medium",
    "enron-routing", 
    "250 emails from Enron employees",
    use_routing=True
)
print(f"   Found {len(ds.filepaths)} files")

# Use search() with the same query content as original compute()
# Original: "Compute the sender, subject, and summary of every email which refers to 
#            the Raptor, Deathstar, Chewco, and/or Fat Boy investments..."
print("\n2. Triggering search (index will build lazily)...")
search_start = time.time()
ds = ds.search(
    "Find emails that refer to the Raptor, Deathstar, Chewco, and/or Fat Boy investments, "
    "excluding emails that quote text from external articles or sources outside of Enron"
)
search_time = time.time() - search_start
print(f"   Search setup complete in {search_time:.2f}s")

# Check if routing happened
if hasattr(ds, 'routed_files') and ds.routed_files:
    print(f"   ✓ Routing active: {len(ds.routed_files)} files selected")
else:
    print("   ✗ No routing (scanning all files)")

# Run the query with same config as original
print("\n3. Executing query...")
config = carnot.QueryProcessorConfig(
    policy=carnot.MaxQuality(),
    progress=False,
)
results = ds.run(config=config)

# Save results
print("\n4. Saving results...")
output_file = "carnot-email-routing-output.csv"
results.to_df().to_csv(output_file, index=False)
print(f"   Results saved to: {output_file}")

# Print summary
total_time = time.time() - start_time
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total execution time: {total_time:.2f}s")
print(f"Results: {len(results.to_df())} records")
print("=" * 80)