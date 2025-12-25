import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Load saved features
X = np.load("data/processed/X_features.npy")
y = np.load("data/processed/y_labels.npy", allow_pickle=True)

# Encode string labels to integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)


#Logistic Regression(Ridge/L2 Regularization)

logreg = LogisticRegression(
    penalty='l2',          # Ridge
    C=1.0,
    solver='lbfgs',
    max_iter=500,
    random_state=42
)

logreg.fit(X_train, y_train)
y_pred_lr = logreg.predict(X_test)

print("=== Logistic Regression ===")
print(classification_report(y_test, y_pred_lr, target_names=le.classes_))

# Support Vector Machine (SVM) 

svm = SVC(
    kernel='linear', 
    C=1.0, 
    probability=True, 
    random_state=42
)

svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

print("=== SVM ===")
print(classification_report(y_test, y_pred_svm, target_names=le.classes_))

joblib.dump(logreg, "models/logreg_model.pkl")
joblib.dump(svm, "models/svm_model.pkl")
joblib.dump(le, "models/label_encoder.pkl")
