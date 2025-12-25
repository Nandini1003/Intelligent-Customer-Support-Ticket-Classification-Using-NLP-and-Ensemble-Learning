from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import re
import string
from collections import Counter
from tensorflow.keras.models import load_model

CONFIDENCE_THRESHOLD = 0.40
TRUSTED_MODELS = ["ANN", "SVM"]

# ------------------------
# Load Pre-trained Models
# ------------------------
logreg_model = joblib.load("models/logreg_model.pkl")
svm_model = joblib.load("models/svm_model.pkl")
kmeans_model = joblib.load("models/kmeans_model.pkl")
tfidf = joblib.load("models/tfidf.pkl")
pca = joblib.load("models/pca.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

ann_model = load_model("models/ann_model.h5")
cnn_model = load_model("models/cnn_model.h5")

# ------------------------
# Initialize Flask App
# ------------------------
app = Flask(__name__)
CORS(app)

# ------------------------
# Preprocessing
# ------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def prepare_features(text):
    cleaned = clean_text(text)
    vector = tfidf.transform([cleaned])
    return pca.transform(vector.toarray())

# ------------------------
# Routes
# ------------------------
@app.route("/predict", methods=["POST"])
def predict_ticket():
    data = request.get_json()
    text = data.get("message", "").strip()

    if not text:
        return jsonify({"error": "No message provided"}), 400

    features = prepare_features(text)

    model_outputs = []
    votes = []

    # Logistic Regression
    lr_pred = logreg_model.predict(features)[0]
    lr_conf = max(logreg_model.predict_proba(features)[0])
    lr_label = label_encoder.inverse_transform([lr_pred])[0]
    model_outputs.append(("LogisticRegression", lr_label, lr_conf))

    # SVM
    svm_pred = svm_model.predict(features)[0]
    svm_conf = max(svm_model.predict_proba(features)[0])
    svm_label = label_encoder.inverse_transform([svm_pred])[0]
    model_outputs.append(("SVM", svm_label, svm_conf))
    votes.append(svm_label)

    # ANN
    ann_probs = ann_model.predict(features)
    ann_pred = np.argmax(ann_probs, axis=1)[0]
    ann_conf = np.max(ann_probs)
    ann_label = label_encoder.inverse_transform([ann_pred])[0]
    model_outputs.append(("ANN", ann_label, ann_conf))
    votes.append(ann_label)

    # CNN (filtered)
    cnn_features = np.expand_dims(features, axis=2)
    cnn_probs = cnn_model.predict(cnn_features)
    cnn_pred = np.argmax(cnn_probs, axis=1)[0]
    cnn_conf = np.max(cnn_probs)
    cnn_label = label_encoder.inverse_transform([cnn_pred])[0]

    if cnn_conf >= CONFIDENCE_THRESHOLD:
        model_outputs.append(("CNN", cnn_label, cnn_conf))
    else:
        model_outputs.append(("CNN", "Low Confidence", cnn_conf))

    # Final decision (ANN + SVM vote)
    final_decision = Counter(votes).most_common(1)[0][0]

    # Clustering
    cluster_label = int(kmeans_model.predict(features)[0])

    return jsonify({
        "input_text": text,
        "final_prediction": final_decision,
        "decision_method": "Majority Vote (ANN + SVM)",
        "cluster_label": cluster_label,
        "model_predictions": [
            {
                "model": m,
                "category": c,
                "confidence": f"{conf*100:.2f}%"
            } for m, c, conf in model_outputs
        ]
    })

# ------------------------
# Run App
# ------------------------
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
