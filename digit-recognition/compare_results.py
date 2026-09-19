import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from data_loader import load_mnist

(X_train, y_train), (X_test, y_test) = load_mnist()

sklearn_preds = np.load("results/sklearn_preds.npy")
nn_preds = np.load("results/nn_preds.npy")

sklearn_acc = accuracy_score(y_test, sklearn_preds)
nn_acc = accuracy_score(y_test, nn_preds)

names = ["Random Forest", "Neural Network"]
accs = [sklearn_acc, nn_acc]

plt.figure(figsize=(8, 5))
bars = plt.bar(names, [a * 100 for a in accs], color=["#5b8db8", "#2c5f8a"])
for bar, a in zip(bars, accs):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             f"{a*100:.2f}%", ha="center", fontweight="bold")
plt.ylabel("Test Accuracy (%)")
plt.title("Model Comparison on MNIST")
plt.ylim(85, 100)
plt.savefig("results/model_comparison.png")
plt.show()
