import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, GlobalMaxPooling1D, Embedding, Flatten
from tensorflow.keras.callbacks import EarlyStopping

# Load features and labels
X = np.load("data/processed/X_features.npy")
y = np.load("data/processed/y_labels.npy", allow_pickle=True)

# Encode string labels to integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# One-hot encode labels for DL
y_onehot = tf.keras.utils.to_categorical(y_encoded)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42, stratify=y_onehot
)

# ----------------------
# 1️⃣ ANN MODEL
# ----------------------
ann_model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(y_onehot.shape[1], activation='softmax')
])

ann_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train ANN
ann_history = ann_model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=20,
    batch_size=32,
    callbacks=[early_stop],
    verbose=2
)

# Evaluate ANN
ann_eval = ann_model.evaluate(X_test, y_test, verbose=0)
print(f"\nANN Test Accuracy: {ann_eval[1]:.4f}")

y_pred_ann = ann_model.predict(X_test)
y_pred_labels_ann = np.argmax(y_pred_ann, axis=1)
y_true_labels = np.argmax(y_test, axis=1)
print("=== ANN Classification Report ===")
print(classification_report(y_true_labels, y_pred_labels_ann, target_names=le.classes_))

# Save ANN
ann_model.save("models/ann_model.h5")

# ----------------------
# 2️⃣ CNN (Experimental – Feature-based)
# ----------------------
# NOTE:
# CNN is used here only for architectural comparison.
# TF-IDF features are not ideal for CNNs as word order is lost.

X_train_cnn = np.expand_dims(X_train, axis=2)
X_test_cnn = np.expand_dims(X_test, axis=2)

cnn_model = Sequential([
    Conv1D(
        filters=64,
        kernel_size=3,
        activation='relu',
        input_shape=(X_train_cnn.shape[1], 1)
    ),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(y_onehot.shape[1], activation='softmax')
])

cnn_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

cnn_model.fit(
    X_train_cnn,
    y_train,
    validation_split=0.1,
    epochs=10,
    batch_size=32,
    callbacks=[early_stop],
    verbose=2
)

# Evaluate CNN
cnn_eval = cnn_model.evaluate(X_test_cnn, y_test, verbose=0)
print(f"\nCNN (Experimental) Test Accuracy: {cnn_eval[1]:.4f}")

y_pred_cnn = cnn_model.predict(X_test_cnn)
y_pred_labels_cnn = np.argmax(y_pred_cnn, axis=1)

print("=== CNN (Experimental) Classification Report ===")
print(classification_report(
    y_true_labels,
    y_pred_labels_cnn,
    target_names=le.classes_
))

cnn_model.save("models/cnn_model.keras")


# Save Label Encoder
joblib.dump(le, "models/label_encoder.pkl")

print("\nAll models and label encoder saved successfully!")
