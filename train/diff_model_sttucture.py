import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import joblib
import matplotlib.pyplot as plt
from train import myModels

import serverChan

TRAIN_DATA = "10w_dataSTA_train_data.pkl"
TRAIN_LABELS = "10w_dataSTA_train_labels.pkl"
TEST_DATA = "10w_dataSTA_test_data.pkl"
TEST_LABELS = "10w_dataSTA_test_labels.pkl"

EPOCHS = 3

def main():

    train_data = joblib.load(TRAIN_DATA)
    train_labels = joblib.load(TRAIN_LABELS)
    test_data = joblib.load(TEST_DATA)
    test_labels = joblib.load(TEST_LABELS)
    print("数据集导入完成")

    print("diff_structures of cnn:")
    acc1 = []

    model = myModels.model_CNN(32)
    history = model.fit(train_data['vectors'], train_labels, epochs=EPOCHS,
                        validation_data=(test_data['vectors'], test_labels))
    acc1.append(history.history['val_accuracy'][EPOCHS-1])

    model = myModels.model_CNN2()
    history = model.fit(train_data['vectors'], train_labels, epochs=EPOCHS,
                        validation_data=(test_data['vectors'], test_labels))
    acc1.append(history.history['val_accuracy'][EPOCHS-1])

    model = myModels.model_CNN3()
    history = model.fit(train_data['vectors'], train_labels, epochs=EPOCHS,
                        validation_data=(test_data['vectors'], test_labels))
    acc1.append(history.history['val_accuracy'][EPOCHS-1])

    model = myModels.model_CNN4(32)
    history = model.fit(train_data['vectors'], train_labels, epochs=EPOCHS,
                        validation_data=(test_data['vectors'], test_labels))
    acc1.append(history.history['val_accuracy'][EPOCHS-1])

    for i in range(4):
        print(f"{i}: {acc1[i]}")

if __name__ == "__main__":
    main()