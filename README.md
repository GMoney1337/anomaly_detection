# 🧠 Anomaly Detection Primer: From Z-Score to Autoencoders

This project accompanies the article *“Anomaly Detection, A Primer”*, and explores how both classical and machine learning methods can be used to detect anomalies in structured data, including real-world software development activity. 

We walk through everything from basic **Z-Score analysis** to **autoencoder-based deep learning**, using both synthetic datasets and real commit histories from the [XZ Utils](https://tukaani.org/xz-backdoor/) repository — a project recently targeted by a high-profile supply chain attack.

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
└── README.md          # This file```
