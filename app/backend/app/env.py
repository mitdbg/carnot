import os

# TODO: modify Carnot execution logic to have deterministic output location(s) that can be specified
IS_LOCAL_ENV = os.getenv("LOCAL_ENV").lower() == "true"
FILESYSTEM = "file" if IS_LOCAL_ENV else "s3"
COMPANY_ENV = os.getenv("COMPANY_ENV", "dev")
BACKEND_ROOT = "/code/backend/"
BASE_DIR = f"s3://carnot-research-{COMPANY_ENV}/"
DATA_DIR = f"s3://carnot-research-{COMPANY_ENV}/data/"
if IS_LOCAL_ENV:
    BASE_DIR = "/carnot/"
    DATA_DIR = "/carnot/data/"
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
