#!/bin/bash

# --- Configuration ---
PROJECT_NAME="carnot"
LOCAL_BASE_DIR="$HOME/.$PROJECT_NAME/carnot"
SECRETS_DIR="compose/secrets"
PG_DATA_DIR="/tmp/pg-data/data"
COMPOSE_FILE="docker-compose.yaml"
OVERRIDE_FILE="docker-compose.local.yaml"
COMPOSE_DIR="compose"

# --- Default Environment Variables ---
export ENV_NAME="dev"
export LOCAL_ENV="true"
export LOCAL_BASE_DIR="$LOCAL_BASE_DIR"
export DOCKERHUB_USERNAME="carnotlocal"
export SETTINGS_ENCRYPTION_KEY="12u1STDIIImTyKtTfkqwPDRCK4dCe65xHfXrPjrTeIU="

# Frontend Build Arguments (Vite variables)
# NOTE: if you want Auth0 to work locally, you will need to set the following variables:
#  - VITE_AUTH0_DOMAIN
#  - VITE_AUTH0_AUDIENCE
#  - VITE_AUTH0_CLIENT_ID
#  - VITE_AUTH0_ORGANIZATION_ID
#  - AUTH0_CLAIMS_NAMESPACE
# mdrusso has the local Auth0 setup for Carnot, ask him for details.
export VITE_API_BASE_URL="http://localhost:8000/api"

# Backend Runtime Environment
export BASE_ORIGINS="http://localhost"

# --- Database Secrets Defaults ---
DB_PASSWORD_DEFAULT="supersecretpassword"
DB_USER_DEFAULT="carnotuser"
DB_NAME_DEFAULT="carnotdb"

# --- Create Directories ---
echo "Creating local data directory: $LOCAL_BASE_DIR"
mkdir -p "$LOCAL_BASE_DIR"

echo "Creating secrets directory: $SECRETS_DIR"
mkdir -p "$SECRETS_DIR"

echo "Creating PostgreSQL data directory: $PG_DATA_DIR"
mkdir -p "$PG_DATA_DIR"

# --- Prepare Context ---
echo "Copying source files to Docker build context..."

# copy relevant source files to build context
rm -rf $COMPOSE_DIR/frontend
rm -rf $COMPOSE_DIR/backend
rm -rf $COMPOSE_DIR/src
rm $COMPOSE_DIR/pyproject.toml
rm $COMPOSE_DIR/README.md
cp -r ../app/frontend $COMPOSE_DIR/frontend
cp -r ../app/backend $COMPOSE_DIR/backend
cp -r ../src $COMPOSE_DIR/src
cp ../pyproject.toml $COMPOSE_DIR/pyproject.toml
cp ../README.md $COMPOSE_DIR/README.md

echo "Cleaning up local artifacts from build context..."
rm -rf $COMPOSE_DIR/secrets/*
rm -rf $COMPOSE_DIR/frontend/node_modules
rm -rf $COMPOSE_DIR/frontend/dist
rm -rf $COMPOSE_DIR/backend/.venv
rm -rf $COMPOSE_DIR/backend/__pycache__
rm -rf $COMPOSE_DIR/backend/src/__pycache__

# --- Create Secrets Files ---
echo "Creating default secrets files in $SECRETS_DIR..."
echo "$DB_PASSWORD_DEFAULT" > "$SECRETS_DIR/db_password.txt"
echo "$DB_USER_DEFAULT" > "$SECRETS_DIR/db_user.txt"
echo "$DB_NAME_DEFAULT" > "$SECRETS_DIR/db_name.txt"

# Change to the directory containing the compose files
pushd $COMPOSE_DIR > /dev/null

# Run Docker Compose
echo "Starting services with Docker Compose..."
docker compose -f $COMPOSE_FILE -f $OVERRIDE_FILE up --build -d
# docker compose -f $COMPOSE_FILE -f $OVERRIDE_FILE build --no-cache && docker compose up -d

# CAPTURE THE EXIT CODE IMMEDIATELY
EXIT_CODE=$?

# Check the status BEFORE leaving the directory
if [ $EXIT_CODE -eq 0 ]; then
    echo "---"
    echo "✅ Services started successfully!"
    echo "Frontend available at: http://localhost"
    echo "Backend available at: http://localhost:8000"
    echo "PostgreSQL available on port 5432 (data in /tmp/pg-data/data on host)"
    echo "Local data mounted to: $LOCAL_BASE_DIR"
    echo "---"
    # Run PS here while we are still in the correct directory
    docker compose -f $COMPOSE_FILE -f $OVERRIDE_FILE ps
else
    echo "---"
    echo "❌ Failed to start services. Exit code: $EXIT_CODE"
    echo "---"
fi

# Change back to the original directory
popd > /dev/null

# Clean up build artifacts only if successful
if [ $EXIT_CODE -eq 0 ]; then
  rm -rf $COMPOSE_DIR/frontend
  rm -rf $COMPOSE_DIR/backend
  rm -rf $COMPOSE_DIR/src
  rm $COMPOSE_DIR/pyproject.toml
  rm $COMPOSE_DIR/README.md
fi

# Exit with the captured code so the shell knows if it failed
exit $EXIT_CODE

