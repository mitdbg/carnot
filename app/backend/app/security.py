import os

from cryptography.fernet import Fernet

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
