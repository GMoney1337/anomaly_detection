import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic file access counts
np.random.seed(1)
accesses = np.concatenate([np.random.randint(20, 40, 100), [2, 120]])

# Compute IQR
Q1 = np.percentile(accesses, 25)
Q3 = np.percentile(accesses, 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
anomalies = np.where((accesses < lower_bound) | (accesses > upper_bound))[0]

# Plot
plt.figure(figsize=(10, 4))
plt.boxplot(accesses, vert=False)
for idx in anomalies:
    plt.scatter(accesses[idx], 1, color='red')
plt.title("IQR Anomaly Detection")
plt.xlabel("Files Accessed")
plt.tight_layout()
plt.show()
