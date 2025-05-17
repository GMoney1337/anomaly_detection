# 🧠 Anomaly Detection: A Primer

This repository supports the article **“Anomaly Detection: A Primer”**, demonstrating both traditional and deep learning-based methods for identifying anomalies in data, including synthetic examples and a real-world case based on Git commit history (using the XZ Utils project).

We walk through everything from basic **Z-Score analysis** to **autoencoder-based deep learning**, using both synthetic datasets and real commit histories from the [XZ Utils](https://tukaani.org/xz-backdoor/) repository, a project recently targeted by a high-profile supply chain attack.

---

## 🔍 What This Repo Covers

- 📏 **Z-Score**: detect outliers in basic numeric distributions  
- 📦 **IQR (Interquartile Range)**: detect anomalous data in skewed or non-normal distributions  
- 🌲 **Isolation Forest**: unsupervised, multivariate anomaly detection using tree ensembles  
- 🤖 **Autoencoders**: deep learning models that learn to reconstruct normal behavior, and fail on anomalies

Each method is explained with code, visualizations, and real-world applications.

---

## 🗂️ Repository Structure
```text
├── classical/         # Traditional methods: Z-score, IQR, Isolation Forest
├── ml/                # Autoencoder scripts for commits + diffs
├── data/              # Git commit metadata, preprocessed features
├── diffs/             # Raw .diff files (from real commits)
├── model/             # Trained autoencoder weights
├── requirements.txt   # Python dependencies
└── README.md          # This file
```
---
## ⚙️Run Classical Methods
```text
# Z-Score Example
python classical/zscore_anomaly_detection.py

# IQR Example
python classical/iqr_anomaly_detection.py

# Isolation Forest Example
python classical/isolation_forest.py
```
#### These scripts simulate common behavioral patterns (like file access or data transfer) and show how basic techniques flag outliers.
---
## 🤖 Run Git Commit Anomaly Detection
```text
# Step 1: Extract commit message embeddings + time features
python ml/extract_commit_features.py

# Step 2: Parse and extract features from .diff files
python ml/extract_diff_features.py

# Step 3: Combine commit + diff features
python ml/merge_features.py

# Step 4: Train the autoencoder
python ml/autoencoder.py

# Step 5: Detect the most anomalous commits
python ml/detect_anomalies.py

# Step 6: Print detailed info on flagged commits
python ml/print_commits.py
```
---
🔍 Example Output
```text
[1192] 315c64c7e18acc59a745b68148188a73e998252b
Author: Jia Tan
Date:   2023-02-01 21:43:33 +0800
Msg:    CI: Update .gitignore for artifacts directory in build-aux.
```
---
## 📁 Data Notes
* All output files (feature vectors, scaler, anomaly indices) are stored in the data/ directory
* Autoencoder weights are saved to model/autoencoder.pt
* Diff files must be extracted ahead of time via:
```text
git log --pretty=format:"%H || %s || %an || %ai" > data/commits_meta.txt
for hash in $(git log --pretty=format:"%H"); do
  git show $hash --patch --pretty="" > diffs/$hash.diff
done
```

## ✒️ Author
Gmoney1337
[LinkedIn](https://www.linkedin.com/in/galenyanofsky/)
