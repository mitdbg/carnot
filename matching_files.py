import os

ROOT_DIR = "data/enron-eval-medium"
KEYWORDS = ["chewco"]

matches = []

for root, _, files in os.walk(ROOT_DIR):
    for fname in files:
        if not fname.lower().endswith(".txt"):
            continue

        path = os.path.join(root, fname)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().lower()
                if any(k in text for k in KEYWORDS):
                    matches.append(path)
        except Exception as e:
            print(f"Failed to read {path}: {e}")

print(f"Found {len(matches)} matching files:\n")
for m in matches:
    print(m)
