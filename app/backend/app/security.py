import json
import os

import boto3
from botocore.exceptions import ClientError
from cryptography.fernet import Fernet

from app.env import COMPANY_ENV

### for local environments ###
# NOTE: only used in local dev/testing; remote uses AWS Secrets Manager
#       perhaps we should update the check below not to throw a ValueError
key = os.getenv("SETTINGS_ENCRYPTION_KEY")

if not key:
    raise ValueError("SETTINGS_ENCRYPTION_KEY environment variable is not set. Cannot start safely.")

cipher_suite = Fernet(key)

def encrypt_value(value: str) -> str:
    """Encrypts a string value."""
    if not value:
        return None
    return cipher_suite.encrypt(value.encode()).decode()

def decrypt_value(token: str) -> str:
    """Decrypts a string token."""
    if not token:
        return None
    return cipher_suite.decrypt(token.encode()).decode()

### for remote environments ###
def get_user_secret_id(user_id: str) -> str:
    """standardize secret naming convention."""
    # NOTE: this replace should be superfluous now that we do it in auth.py, but keeping for safety
    user_id = user_id.replace("|", "-")
    return f"carnot/{COMPANY_ENV}/{user_id}"

async def save_user_secrets(user_id: str, secret_dict: dict):
    client = boto3.client('secretsmanager', region_name='us-east-1')
    secret_id = get_user_secret_id(user_id)
    secret_string = json.dumps(secret_dict)
    try:
        # check if secret exists
        client.describe_secret(SecretId=secret_id)
        client.put_secret_value(SecretId=secret_id, SecretString=secret_string)
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            # create secret if it doesn't exist
            client.create_secret(Name=secret_id, SecretString=secret_string)
        else:
            raise e

async def get_user_secrets(user_id: str) -> dict:
    client = boto3.client('secretsmanager', region_name='us-east-1')
    secret_id = get_user_secret_id(user_id)
    try:
        response = client.get_secret_value(SecretId=secret_id)
        return json.loads(response['SecretString'])
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            return {}
        raise e
