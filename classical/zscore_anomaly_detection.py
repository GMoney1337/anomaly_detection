import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic login duration data
np.random.seed(0)
durations = np.concatenate([np.random.normal(10, 2, 100), [25, 30]])

# Compute z-scores
mean = np.mean(durations)
std = np.std(durations)
z_scores = (durations - mean) / std

# Flag anomalies
threshold = 3
anomalies = np.where(np.abs(z_scores) > threshold)[0]

# Plot
plt.figure(figsize=(10, 4))
plt.hist(durations, bins=30, alpha=0.7, label="Login durations")
for idx in anomalies:
    plt.axvline(durations[idx], color='red', linestyle='--')
plt.title("Z-Score Anomaly Detection")
plt.xlabel("Duration (seconds)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
