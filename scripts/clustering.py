import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

# Load features
X = np.load("data/processed/X_features.npy")
df = pd.read_csv("data/processed/clean_tickets.csv")

# Find optimal clusters (optional)
sil_score = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    sil_score.append(silhouette_score(X, labels))
print("Silhouette scores for k=2..9:", sil_score)

# Choose k (example: 6)
k = 6
kmeans = KMeans(n_clusters=k, random_state=42)
cluster_labels = kmeans.fit_predict(X)
df["cluster"] = cluster_labels

# Explore clusters
for i in range(k):
    print(f"\n--- Cluster {i} Sample Tickets ---")
    print(df[df["cluster"]==i]["customer_message"].head(5))

# Save KMeans model
joblib.dump(kmeans, "models/kmeans_model.pkl")
print("Clustering completed and model saved!")
