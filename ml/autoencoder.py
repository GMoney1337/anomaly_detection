import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Load and normalize data
data = np.load("data/merged_vectors.npy")

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Save scaler for use during inference
np.save("data/scaler_mean.npy", scaler.mean_)
np.save("data/scaler_scale.npy", scaler.scale_)

# Convert to PyTorch tensors
X = torch.tensor(data_scaled, dtype=torch.float32)
dataset = TensorDataset(X)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

input_dim = X.shape[1]
encoding_dim = 64  # bottleneck

# Define Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Train loop
EPOCHS = 50
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in loader:
        inputs = batch[0]
        outputs = model(inputs)
        loss = criterion(outputs, inputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.6f}")

# Save model weights
torch.save(model.state_dict(), "model/autoencoder.pt")
print("[✓] Model saved to model/autoencoder.pt")
