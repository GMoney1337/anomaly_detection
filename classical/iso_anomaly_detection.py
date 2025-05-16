import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Simulate file transfer data (time, size)
np.random.seed(42)
normal = np.random.multivariate_normal([13, 300], [[4, 50], [50, 10000]], 300)
anomalous = np.array([[2, 8000], [22, 9000], [3, 10000]])
X = np.vstack([normal, anomalous])

# Fit Isolation Forest
clf = IsolationForest(contamination=0.01, random_state=42)
labels = clf.fit_predict(X)

# Plot
plt.figure(figsize=(10, 6))
colors = ['red' if l == -1 else 'blue' for l in labels]
plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolor='k')
plt.title("Isolation Forest: File Transfer Time vs File Size")
plt.xlabel("Transfer Time (hour)")
plt.ylabel("File Size (MB)")
plt.grid(True)
plt.tight_layout()
plt.show()
