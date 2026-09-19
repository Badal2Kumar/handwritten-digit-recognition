import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, classification_report
from data_loader import load_mnist

(X_train, y_train), (X_test, y_test) = load_mnist()

X_train = X_train.reshape(-1, 784).astype("float32") / 255.0
X_test = X_test.reshape(-1, 784).astype("float32") / 255.0

y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

model = Sequential([
    Dense(512, activation="relu", input_shape=(784,)),
    Dropout(0.2),
    Dense(256, activation="relu"),
    Dropout(0.2),
    Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

t0 = time.time()
history = model.fit(
    X_train, y_train_cat,
    epochs=15,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)
print(f"Training took {time.time() - t0:.1f}s")

test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"Neural Network test accuracy: {test_acc*100:.2f}%")

y_pred = np.argmax(model.predict(X_test), axis=1)
print(classification_report(y_test, y_pred))

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax2.plot(history.history["accuracy"], label="train")
ax2.plot(history.history["val_accuracy"], label="validation")
ax2.set_title("Accuracy per epoch")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.legend()

ax2.plot(history.history["loss"], label="train")
ax2.plot(history.history["val_loss"], label="validation")
ax2.set_title("Loss per epoch")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()
plt.savefig("results/nn_training_curves.png")
plt.show()

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
indices = np.random.choice(len(X_test), 10, replace=False)
X_show = X_test.reshape(-1, 28, 28)
for i, idx in enumerate(indices):
    axes.flat[i].imshow(X_show[idx], cmap="gray")
    color = "green" if y_pred[idx] == y_test[idx] else "red"
    axes.flat[i].set_title(f"Pred: {y_pred[idx]} | True: {y_test[idx]}", color=color)
    axes.flat[i].axis("off")
plt.suptitle("Neural network predictions")
plt.tight_layout()
plt.savefig("results/nn_predictions.png")
plt.show()

model.save("results/digit_model.keras")
np.save("results/nn_preds.npy", y_pred)
