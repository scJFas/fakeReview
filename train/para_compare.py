import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import joblib
import matplotlib.pyplot as plt

import model_cnn
import serverChan

EPOCHS = 7

PARA_1 = [5,10,25,29,32,40,45,50,55,60,64]

TRAIN_VECTORS = "first_train_vectors.pkl"
TRAIN_LABELS = "first_train_labels.pkl"
TEST_VECTORS = "first_test_vector.pkl"
TEST_LABELS = "first_test_labels.pkl"

def main():
    #model = model_cnn.model_CNN()
    train_vectors = joblib.load(TRAIN_VECTORS)
    train_labels = joblib.load(TRAIN_LABELS)
    test_vectors = joblib.load(TEST_VECTORS)
    test_labels = joblib.load(TEST_LABELS)
    print("数据集导入完成")
    historys = []
    for para in PARA_1:
        model = model_cnn.model_CNN(para)
        history = model.fit(train_vectors, train_labels, epochs=EPOCHS, validation_data=(test_vectors, test_labels))
        historys.append(history)

    # plt.plot(history.history['accuracy'], label='accuracy')
    # plt.plot(history.history['val_accuracy'], label='val_accuracy')
    for i in range(len(PARA_1)):
        plt.plot(historys[i].history['val_accuracy'], label=f"filter={PARA_1[i]}")

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.7, 0.85])
    plt.legend(loc='lower right')
    plt.show()

    serverChan.sendMessage("参数比较训练完成")
    return 0

if __name__ == "__main__":
    main()