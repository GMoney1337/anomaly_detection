import os
import numpy as np
import datetime
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')
commit_data = []

# Step 1: Load commit messages and timestamps
with open("commits_meta.txt", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(" || ")
        if len(parts) != 4:
            continue
        commit_hash, msg, author, timestamp = parts
        # Strip timezone from timestamp (drop +0300, etc.)
        timestamp = timestamp.split(" ")[0] + " " + timestamp.split(" ")[1]
        commit_data.append({
            "msg": msg,
            "timestamp": timestamp
        })

# Step 2: Feature extraction
def extract_features(commit):
    msg_embed = model.encode(commit['msg'])  # 384D
    try:
        dt = datetime.datetime.strptime(commit['timestamp'], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        dt = datetime.datetime.strptime(commit['timestamp'], "%Y-%m-%d %H:%M")
    hour = dt.hour
    day_of_week = dt.weekday()
    time_feats = np.array([hour, day_of_week])
    return np.concatenate([msg_embed, time_feats])

# Step 3: Create dataset
feature_vectors = np.array([extract_features(c) for c in tqdm(commit_data)])

# Step 4: Save to .npy
np.save("commit_vectors_simple.npy", feature_vectors)
