import os
import numpy as np
import math
from tqdm import tqdm

DIFF_DIR = "diffs"
OUTPUT_FILE = "data/diff_features.npy"

def is_valid_diff(content):
    return "diff --git" in content and any(l.startswith("+") for l in content.splitlines())

def shannon_entropy(text):
    if not text:
        return 0.0
    probs = [text.count(c) / len(text) for c in set(text)]
    return -sum(p * math.log2(p) for p in probs)

diff_features = {}

print("[+] Parsing diffs...")
for filename in tqdm(os.listdir(DIFF_DIR)):
    if not filename.endswith(".diff"):
        continue
    commit_hash = filename[:-5]  # remove ".diff"
    path = os.path.join(DIFF_DIR, filename)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    if not is_valid_diff(content):
        continue

    lines = content.splitlines()
    added_lines = [l[1:] for l in lines if l.startswith('+') and not l.startswith('+++')]
    removed_lines = [l[1:] for l in lines if l.startswith('-') and not l.startswith('---')]

    added = len(added_lines)
    removed = len(removed_lines)
    file_count = sum(1 for l in lines if l.startswith("diff --git"))

    added_text = ''.join(added_lines)
    entropy = shannon_entropy(added_text)

    diff_features[commit_hash] = np.array([added, removed, file_count, entropy])

# Save output
np.save(OUTPUT_FILE, diff_features)
print(f"[✓] Saved diff features to {OUTPUT_FILE}")
