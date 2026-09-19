import os
import gzip
import urllib.request
import numpy as np

BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"


def download(filename):
    filepath = os.path.join("data", filename)
    if not os.path.exists(filepath):
        os.makedirs("data", exist_ok=True)
        print(f"Downloading {filename}...")
    urllib.request.urlretrieve(BASE_URL + filename, filepath)
    return filepath


def load_images(filename):
    with gzip.open(download(filename), "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)


def load_labels(filename):
    with gzip.open(download(filename), "rb") as f:
        return np.frombuffer(f.read(), np.uint8, offset=8)


def load_mnist():
    X_train = load_images("train-images-idx3-ubyte.gz")
    y_train = load_labels("train-labels-idx1-ubyte.gz")
    X_test = load_images("t10k-images-idx3-ubyte.gz")
    y_test = load_labels("t10k-labels-idx1-ubyte.gz")
    return (X_train, y_train), (X_test, y_test)


if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = load_mnist()
    print("Training images:", X_train.shape)
    print("Test images:", X_test.shape)
    print("Pixel range:", X_train.min(), "-", X_train.max())
