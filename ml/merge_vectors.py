import numpy as np

# Load data
commit_vectors = np.load("data/commit_vectors.npy")
commit_ids = np.load("data/commit_ids.npy")
diff_features = np.load("data/diff_features.npy", allow_pickle=True).item()

merged_vectors = []

# Align and combine features
for i, commit_id in enumerate(commit_ids):
    if commit_id in diff_features:
        combined = np.concatenate([commit_vectors[i], diff_features[commit_id]])
        merged_vectors.append(combined)

merged_vectors = np.array(merged_vectors)

# Save merged dataset
np.save("data/merged_vectors.npy", merged_vectors)
print(f"[✓] Merged {len(merged_vectors)} vectors and saved to data/merged_vectors.npy")
