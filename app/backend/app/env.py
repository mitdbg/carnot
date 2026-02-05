import os

IS_LOCAL_ENV = os.getenv("LOCAL_ENV", "").lower() == "true"
FILESYSTEM = "file" if IS_LOCAL_ENV else "s3"
REGION_NAME = os.getenv("AWS_REGION", "us-east-1")
COMPANY_ENV = os.getenv("COMPANY_ENV", "dev")
BACKEND_ROOT = "/code/"
BASE_DIR = f"s3://carnot-research-{COMPANY_ENV}/" 
DATA_DIR = f"s3://carnot-research-{COMPANY_ENV}/data/"
SHARED_DATA_DIR = f"s3://carnot-research-{COMPANY_ENV}/shared/"
if IS_LOCAL_ENV:
    BASE_DIR = "/carnot/"
    DATA_DIR = "/carnot/data/"
    SHARED_DATA_DIR = "/carnot/shared/"
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(SHARED_DATA_DIR, exist_ok=True)
