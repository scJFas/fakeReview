import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import joblib
import matplotlib.pyplot as plt

from train import myModels
import serverChan

EPOCHS = 10

TRAIN_DATA = "1w_train_data.pkl"
TRAIN_LABELS = "1w_train_labels.pkl"
TEST_DATA = "1w_test_data.pkl"
TEST_LABELS = "1w_test_labels.pkl"

SAVE = False
SAVE_MODEL_NAME = "10w_model1.h5"

def main():
    model = myModels.model_CNN()
    train_data = joblib.load(TRAIN_DATA)
    train_labels = joblib.load(TRAIN_LABELS)
    test_data = joblib.load(TEST_DATA)
    test_labels = joblib.load(TEST_LABELS)
    print("数据集导入完成")

    # history = model.fit([train_data['vectors'], train_data['user_actions'], train_data['shop_actions']], train_labels, epochs=EPOCHS,
    #                     validation_data=([test_data['vectors'], test_data['user_actions'], test_data['shop_actions']], test_labels))

    history = model.fit(train_data['vectors'], train_labels, epochs=EPOCHS,  validation_data=(test_data['vectors'], test_labels))

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

    serverChan.sendMessage("模型训练完成")

    if(SAVE):
        model.save(SAVE_MODEL_NAME)
    return 0

if __name__ == "__main__":
    main()