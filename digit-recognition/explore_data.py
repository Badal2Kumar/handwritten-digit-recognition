import numpy as np
import matplotlib.pyplot as plt
import os
from data_loader import load_mnist

os.makedirs("results", exist_ok=True)

(X_train, y_train), (X_test, y_test) = load_mnist()

print("Training images:", X_train.shape)
print("Test images:", X_test.shape)
print("Pixel range:", X_train.min(), "-", X_train.max())

labels, counts = np.unique(y_train, return_counts=True)
plt.bar(labels, counts, color="steelblue")
plt.title("Digit distribution in training set")
plt.xlabel("Digit")
plt.ylabel("Count")
plt.savefig("results/class_distribution.png")
plt.show()

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i], cmap="gray")
    ax.set_title(f"Label: {y_train[i]}")
    ax.axis("off")
plt.suptitle("Sample MNIST digits")
plt.tight_layout()
plt.savefig("results/sample_digits.png")
plt.show()
