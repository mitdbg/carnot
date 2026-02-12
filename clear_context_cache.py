#!/usr/bin/env python3
"""
Utility script to clear the Carnot context cache.

This script removes all pickled context files and clears the ChromaDB collection.
Use this if you encounter persistent errors with corrupted contexts.
"""
import os
import shutil

import chromadb

from carnot.constants import PZ_DIR


def clear_cache():
    """Clear all cached contexts and the ChromaDB index."""
    print("Clearing Carnot context cache...")
    
    # Remove context files
    context_dir = os.path.join(PZ_DIR, "contexts")
    if os.path.exists(context_dir):
        file_count = len([f for f in os.listdir(context_dir) if f.endswith('.pkl')])
        for filename in os.listdir(context_dir):
            file_path = os.path.join(context_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Warning: Could not remove {file_path}: {e}")
        print(f"✓ Removed {file_count} context file(s)")
    else:
        print("✓ No context directory found")
    
    # Clear ChromaDB collection
    chroma_dir = os.path.join(PZ_DIR, "chroma")
    if os.path.exists(chroma_dir):
        try:
            chroma_client = chromadb.PersistentClient(chroma_dir)
            try:
                chroma_client.delete_collection("contexts")
                print("✓ Cleared ChromaDB contexts collection")
            except chromadb.errors.NotFoundError:
                print("✓ No ChromaDB collection to clear")
        except Exception as e:
            print(f"Warning: Could not clear ChromaDB: {e}")
    else:
        print("✓ No ChromaDB directory found")
    
    print("\nCache cleared successfully!")


if __name__ == "__main__":
    import sys
    
    # Ask for confirmation
    response = input("This will delete all cached contexts. Continue? (y/N): ")
    if response.lower() != 'y':
        print("Cancelled.")
        sys.exit(0)
    
    clear_cache()


