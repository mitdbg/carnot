"""
Downloads and extracts the Enron email dataset used for tests.

The zip file is hosted at:
  https://palimpzest-workloads.s3.us-east-1.amazonaws.com/emails.zip

Running this script will create:
  tests/pytest/data/emails/
"""

import urllib.request
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
ZIP_URL = "https://palimpzest-workloads.s3.us-east-1.amazonaws.com/emails.zip"
ZIP_PATH = DATA_DIR / "emails.zip"
DEST_DIR = DATA_DIR / "emails"


def main():
    if DEST_DIR.exists():
        print(f"Dataset already exists at {DEST_DIR}. Skipping download.")
        return

    print(f"Downloading {ZIP_URL} ...")
    urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
    print(f"Saved zip to {ZIP_PATH}")

    print("Extracting ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATA_DIR)
    print(f"Extracted to {DEST_DIR}")

    ZIP_PATH.unlink()
    print("Removed zip file.")


if __name__ == "__main__":
    main()
