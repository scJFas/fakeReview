import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.externals import joblib
import matplotlib.pyplot as plt

from train.model_cnn import model_CNN

EPOCHS = 10

TRAIN_VECTORS = ""
TRAIN_LABELS = ""
TEST_VECTORS = ""
TEST_LABELS = ""

def main():
    model = model_CNN()
    train_vectors = joblib.load(TRAIN_VECTORS)
    train_labels = joblib.load(TRAIN_LABELS)
    test_vectors = joblib.load(TEST_VECTORS)
    test_labels = joblib.load(TEST_LABELS)
    history = model.fit(train_vectors, train_labels, epochs=EPOCHS, validation_data=(test_vectors, test_labels))

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    main()