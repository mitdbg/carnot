import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import init_db
from app.routes import config, conversations, datasets, files, query, search, settings, workspaces

# compute allowed origins for CORS
allowed_origins = ["http://localhost", "http://localhost:80"]
BASE_ORIGINS = os.getenv("BASE_ORIGINS")
if BASE_ORIGINS is not None:
    for origin in BASE_ORIGINS.split(","):
        allowed_origins.append(origin.strip())

# Initialize database
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

app = FastAPI(title="Carnot Web API", lifespan=lifespan)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(config.router, prefix="/api/config", tags=["config"])
app.include_router(files.router, prefix="/api/files", tags=["files"])
app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])
app.include_router(search.router, prefix="/api/search", tags=["search"])
app.include_router(query.router, prefix="/api/query", tags=["query"])
app.include_router(conversations.router, prefix="/api/conversations", tags=["conversations"])
app.include_router(workspaces.router, prefix="/api/workspaces", tags=["workspaces"])
app.include_router(settings.router, prefix="/api/settings", tags=["settings"])

@app.get("/")
async def root():
    return {"message": "Carnot Web API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

