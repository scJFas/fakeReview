import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import joblib
import matplotlib.pyplot as plt

from train import myModels
import serverChan

EPOCHS = 10

TRAIN_VECTORS = "10w_train_vectors.pkl"
TRAIN_LABELS = "10w_train_labels.pkl"
TEST_VECTORS = "10w_test_vector.pkl"
TEST_LABELS = "10w_test_labels.pkl"

SAVE_MODEL_NAME = "10w_model1.h5"

def main():
    model = myModels.model_CNN()
    train_vectors = joblib.load(TRAIN_VECTORS)
    train_labels = joblib.load(TRAIN_LABELS)
    test_vectors = joblib.load(TEST_VECTORS)
    test_labels = joblib.load(TEST_LABELS)
    print("数据集导入完成")

    history = model.fit(train_vectors, train_labels, epochs=EPOCHS, validation_data=(test_vectors, test_labels))

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

    serverChan.sendMessage("模型训练完成")

    model.save(SAVE_MODEL_NAME)
    return 0

if __name__ == "__main__":
    main()