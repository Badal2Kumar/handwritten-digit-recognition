import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from data_loader import load_mnist

(X_train, y_train), (X_test, y_test) = load_mnist()

X_train_flat = X_train.reshape(len(X_train), -1) / 255.0
X_test_flat = X_test.reshape(len(X_test), -1) / 255.0

results = {}

print("Training Logistic Regression...")
t0 = time.time()
log_reg = LogisticRegression(max_iter=1000, n_jobs=-1)
log_reg.fit(X_train_flat, y_train)
log_pred = log_reg.predict(X_test_flat)
log_acc = accuracy_score(y_test, log_pred)
log_time = time.time() - t0
results["Logistic Regression"] = (log_acc, log_time)
print(f"Done in {log_time:.1f}s, accuracy: {log_acc*100:.2f}%")

print("Training Random Forest...")
t0 = time.time()
rf = RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=42)
rf.fit(X_train_flat, y_train)
rf_pred = rf.predict(X_test_flat)
rf_acc = accuracy_score(y_test, rf_pred)
rf_time = time.time() - t0
results["Random Forest"] = (rf_acc, rf_time)
print(f"Done in {rf_time:.1f}s, accuracy: {rf_acc*100:.2f}%")

print("\nClassification Report (Random Forest):")
print(classification_report(y_test, rf_pred))

cm = confusion_matrix(y_test, rf_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("results/cm_rf.png")
plt.show()

np.save("results/sklearn_preds.npy", rf_pred)

for name, (acc, t) in results.items():
    print(f"{name}: {acc*100:.2f}% accuracy, {t:.1f}s training time")
