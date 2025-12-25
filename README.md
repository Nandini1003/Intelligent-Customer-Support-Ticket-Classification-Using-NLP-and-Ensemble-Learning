# Customer Support Ticket Classifier

## Project Overview

The **Customer Support Ticket Classifier** is an end-to-end NLP-based machine learning system designed to automatically categorize customer support tickets into predefined classes such as *Billing, Refund, Technical Issue, Account, Delivery/Service,* and *General Inquiry*. The system leverages an **ensemble of classical ML and deep learning models**, combined with confidence-aware decision logic and a real-time web interface.

This project demonstrates **production-style ML thinking**, not just model training — including preprocessing pipelines, model comparison, confidence thresholds, explainable outputs, and deployment readiness.

---

## Business Problem

Customer support teams handle large volumes of unstructured text tickets daily. Manual categorization is:

* Time-consuming
* Error-prone
* Not scalable

This system automates ticket classification to:

* Reduce response time
* Improve routing accuracy
* Enable data-driven insights through clustering



##  Solution Architecture

### Models Used

* **Logistic Regression (TF-IDF + PCA)** – Baseline, interpretable model
* **Support Vector Machine (SVM)** – High-accuracy linear classifier
* **Artificial Neural Network (ANN)** – Dense neural network for non-linear patterns
* **Convolutional Neural Network (CNN)** – Experimental deep learning model
* **KMeans Clustering** – Unsupervised grouping for pattern discovery

### Decision Strategy

* Only **high-confidence models (≥ 70%)** participate in final prediction
* Final output is generated using **majority voting (ANN + SVM)**
* CNN is treated as an **experimental, non-voting model** for research comparison


##  Data Preprocessing Pipeline

* Text cleaning (lowercasing, punctuation removal, whitespace normalization)
* TF-IDF vectorization
* Dimensionality reduction using PCA
* Label encoding for supervised learning
* One-hot encoding for deep learning models



## Model Performance (Highlights)

* **SVM Accuracy:** ~99%
* **ANN Accuracy:** ~99%
* Logistic Regression performs strongly as a baseline
* CNN shows lower confidence and is intentionally excluded from production decisions

> Performance metrics are used selectively to ensure robustness and avoid overfitting bias.

##  Web Application (Demo)

### Features

* Clean, recruiter-friendly UI
* Real-time ticket classification
* Confidence bars for each model
* Clear distinction between **participating** and **experimental** models
* Explanation of decision logic

### Tech Stack

* **Backend:** Flask, TensorFlow, scikit-learn
* **Frontend:** HTML, CSS, JavaScript
* **API:** REST-based /predict endpoint


## Project Structure


project-root/
│
├── app.py                  # Flask backend
├── models/                 # Trained models & encoders
│   ├── ann_model.h5
│   ├── cnn_model.h5
│   ├── svm_model.pkl
│   ├── logreg_model.pkl
│   ├── label_encoder.pkl
│   ├── tfidf.pkl
│   ├── pca.pkl
│   └── kmeans_model.pkl
│
├── data/processed/          # Feature arrays
│   ├── X_features.npy
│   └── y_labels.npy
│
├── templates/
│   └── index.html           # UI
│
├── static/                  # CSS / JS
├── README.md
└── requirements.txt

## 🚀 How to Run the Project

### 1️. Install Dependencies
### 2️. Start the Flask Server
### 3. Open the Web App


## Key Design Decisions

* CNN retained for **architectural experimentation**, not blind accuracy
* Confidence-based filtering avoids unreliable predictions
* Ensemble approach improves robustness over single-model systems
* UI explicitly communicates trust and transparency


## Future Improvements

* Replace CNN with Transformer-based models (BERT)
* Add active learning for low-confidence tickets
* Deploy using Docker + cloud hosting
* Add monitoring for model drift


> This project is intentionally designed to reflect real-world ML engineering practices rather than academic-only modeling.
