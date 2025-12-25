import pandas as pd
import numpy as np
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# Paths
INPUT_PATH = "data/processed/clean_tickets.csv"
TFIDF_PATH = "models/tfidf.pkl"
PCA_PATH = "models/pca.pkl"
X_FEATURES_PATH = "data/processed/X_features.npy"
Y_LABELS_PATH = "data/processed/y_labels.npy"


def main():
    # Load cleaned data
    df = pd.read_csv(INPUT_PATH)

    X_text = df["clean_message"]
    y = df["category"]

    # TF-IDF
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=5
    )
    X_tfidf = tfidf.fit_transform(X_text)

    # PCA
    pca = PCA(n_components=300, random_state=42)
    X_pca = pca.fit_transform(X_tfidf.toarray())

    # Save models
    joblib.dump(tfidf, TFIDF_PATH)
    joblib.dump(pca, PCA_PATH)

    # Save features for training
    np.save(X_FEATURES_PATH, X_pca)
    np.save(Y_LABELS_PATH, y)

    print("Feature engineering completed successfully.")
    print("X shape:", X_pca.shape)
    print("y shape:", y.shape)


if __name__ == "__main__":
    main()
