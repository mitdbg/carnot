#!/bin/bash

for r in 0 1 2 3; do
  tmux new -d -s chroma_r$r \
    "SKUNK_CHROMADB_DIR=$HOME/chromadb-shards/r$r SKUNK_CHROMA_SERVER_PORT=$((8000+r)) \
     ./scripts/run_chroma_server.sh qwen-biogen-0.6b_r$r 2>&1 | tee $HOME/chroma_r$r.log"
done
# watch them warm (each logs '[warm] loaded HNSW index for qwen-biogen-0.6b_rN'):
tail -f ~/chroma_r0.log

