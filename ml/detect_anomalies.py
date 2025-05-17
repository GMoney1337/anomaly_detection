import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# Load merged commit data
X = np.load("data/merged_vectors.npy")

# Load scaler and normalize
mean = np.load("data/scaler_mean.npy")
scale = np.load("data/scaler_scale.npy")
scaler = StandardScaler()
scaler.mean_ = mean
scaler.scale_ = scale
X_scaled = scaler.transform(X)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# Auto-detect input dim from data
INPUT_DIM = X.shape[1]
ENCODING_DIM = 64

# Define the same model structure as used during training
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, ENCODING_DIM)
        )
        self.decoder = nn.Sequential(
            nn.Linear(ENCODING_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, INPUT_DIM)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# Load model weights
model = Autoencoder()
model.load_state_dict(torch.load("model/autoencoder.pt"))
model.eval()

# Compute reconstruction errors
with torch.no_grad():
    recon = model(X_tensor)
    mse = ((X_tensor - recon) ** 2).mean(dim=1).numpy()

# Determine anomaly threshold
threshold = np.percentile(mse, 99)  # Top 1%
anomalies = np.where(mse >= threshold)[0]

# Save anomalies for review
np.save("data/anomaly_indices.npy", anomalies)

# Output results
print(f"Total commits: {len(X)}")
print(f"Anomalies detected (top 1%): {len(anomalies)}")
print("Indices of most anomalous commits:")
print(anomalies)
print("Corresponding MSE values:")
print(mse[anomalies])
