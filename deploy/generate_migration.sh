#!/bin/bash
set -e

# --- Argument Validation ---
if [ -z "$1" ]; then
  echo "❌ Error: Missing migration message."
  echo "Usage: $0 \"Your Migration Message Here\""
  exit 1
fi

MIGRATION_MESSAGE="$1"

# start the local stack (in the background)
echo "🚀 Starting local environment..."
./start_local.sh

# wait for backend to be ready
echo "⏳ Waiting for backend container..."
sleep 10
CONTAINER_NAME=compose-web-backend-1

if [ -z "$CONTAINER_NAME" ]; then
  echo "❌ Could not find backend container. Check 'docker compose ps'."
  exit 1
fi

# generate the revision inside the container (we use the container because it has the /run/secrets/ files mounted)
echo "🛠 Generating Alembic revision inside container with message: \"$MIGRATION_MESSAGE\""
docker exec -it $CONTAINER_NAME alembic revision --autogenerate -m "$MIGRATION_MESSAGE"

# copy the migration file back to host
echo "📂 Copying migration file to local host..."
# Note: This path assumes your docker-compose mounts '.' to '/app' 
# or that you want to copy from the container's workdir.
docker cp $CONTAINER_NAME:/code/alembic/versions/. ../app/backend/alembic/versions/

echo "✅ Success! A new migration file is in app/backend/alembic/versions/"
echo "👉 Review the file, then: git add app/backend/alembic/versions/ && git commit ..."
