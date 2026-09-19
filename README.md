# Handwritten Digit Recognition (MNIST)
A machine learning project that recognizes handwritten digits (0–9) from the
MNIST dataset using multiple models: Logistic Regression and Random Forest
(scikit-learn) and a Neural Network (TensorFlow/Keras), with full evaluation
and visual comparison of results.

## Table of Contents
- [Dataset](#dataset)
- [Libraries Used](#libraries-used)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Workflow](#workflow)
- [Results](#results)
- [Screenshots](#screenshots)
- [External Resources](#external-resources)

## Dataset
- **MNIST database of handwritten digits**
- Source: http://yann.lecun.com/exdb/mnist/ (official mirror used by the loader)
- 70,000 grayscale images of 28×28 pixels
  - 60,000 training images
  - 10,000 test images
- Each image is a handwritten digit (0–9), labels 0 through 9
- Pixel values range from 0 (black) to 255 (white)
- The dataset is **downloaded automatically** by `data_loader.py` and cached
  in the `data/` folder (not included in this repository)

## Libraries Used
| Library | Purpose |

| NumPy | Array operations, data handling, normalization |
| Matplotlib | Visualization (charts, confusion matrix, predictions) |
| scikit-learn | Logistic Regression, Random Forest, metrics |
| TensorFlow / Keras | Neural network model building and training |
| Python 3.9+ | Programming language |
