import os
import numpy as np
import datetime
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
commit_ids = []
commit_vectors = []

# Step 1: Load commit messages and timestamps
with open("data/commits_meta.txt", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(" || ")
        if len(parts) != 4:
            continue
        commit_hash, msg, author, timestamp = parts
        timestamp = timestamp.split(" ")[0] + " " + timestamp.split(" ")[1]
        
        # Extract features
        msg_embed = model.encode(msg)  # 384D
        try:
            dt = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            dt = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M")
        hour = dt.hour
        day_of_week = dt.weekday()
        time_feats = np.array([hour, day_of_week])
        
        full_vector = np.concatenate([msg_embed, time_feats])
        commit_ids.append(commit_hash)
        commit_vectors.append(full_vector)

# Step 2: Save vectors and commit IDs
np.save("data/commit_vectors.npy", np.array(commit_vectors))
np.save("data/commit_ids.npy", np.array(commit_ids))

print(f"[✓] Saved {len(commit_vectors)} vectors to data/commit_vectors.npy")
