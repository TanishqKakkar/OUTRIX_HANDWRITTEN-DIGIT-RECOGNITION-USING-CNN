from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np
import time
import joblib   # for saving/loading models

# 1. Load Data
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype('int')

# 2. Preprocess Data
X = X / 255.0
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# ------------------ Logistic Regression ------------------
print("\n=== Logistic Regression ===")
start = time.time()
log_reg = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto')
log_reg.fit(X_train, y_train)
log_time = time.time() - start
log_pred = log_reg.predict(X_test)
log_acc = accuracy_score(y_test, log_pred)
print(f"Accuracy: {log_acc:.4f} (Time: {log_time:.2f} sec)")
print("Classification Report:")
print(classification_report(y_test, log_pred))
# Save model
joblib.dump(log_reg, "logistic_regression_mnist.pkl")
print("Saved: logistic_regression_mnist.pkl")

# ------------------ Support Vector Machine (SVM) ------------------
print("\n=== Support Vector Machine (subset) ===")
# Use subset for speed
subset_train = 10000
subset_test = 2000
start = time.time()
svm_clf = SVC(kernel='rbf', gamma='scale', probability=True)
svm_clf.fit(X_train[:subset_train], y_train[:subset_train])
svm_time = time.time() - start
svm_pred = svm_clf.predict(X_test[:subset_test])
svm_acc = accuracy_score(y_test[:subset_test], svm_pred)
print(f"Accuracy (subset): {svm_acc:.4f} (Time: {svm_time:.2f} sec)")
print("Classification Report (subset):")
print(classification_report(y_test[:subset_test], svm_pred))
# Save model
joblib.dump(svm_clf, "svm_mnist_subset.pkl")
print("Saved: svm_mnist_subset.pkl")

# ------------------ Random Forest ------------------
print("\n=== Random Forest ===")
start = time.time()
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_clf.fit(X_train, y_train)
rf_time = time.time() - start
rf_pred = rf_clf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"Accuracy: {rf_acc:.4f} (Time: {rf_time:.2f} sec)")
print("Classification Report:")
print(classification_report(y_test, rf_pred))
# Save model
joblib.dump(rf_clf, "random_forest_mnist.pkl")
print("Saved: random_forest_mnist.pkl")

# ------------------ Sample Visualization ------------------
def show_sample_prediction(index, model, model_name):
    plt.imshow(X_test[index].reshape(28, 28), cmap='gray')
    plt.title(f"{model_name} Prediction: {model.predict([X_test[index]])[0]} "
              f"| True: {y_test[index]}")
    plt.axis('off')
    plt.show()

print("\nVisualizing a few Random Forest predictions...")
for idx in [0, 1, 2]:
    show_sample_prediction(idx, rf_clf, "Random Forest")

print("\nAll models saved successfully.")
