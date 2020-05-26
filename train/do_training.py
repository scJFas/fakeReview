import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import joblib
import matplotlib.pyplot as plt
from train import myModels
import serverChan

EPOCHS = 3

from preProcess.data_standardization import normalization

TRAIN_DATA = "10w_dataSTA_train_data.pkl"
TRAIN_LABELS = "10w_dataSTA_train_labels.pkl"
TEST_DATA = "10w_dataSTA_test_data.pkl"
TEST_LABELS = "10w_dataSTA_test_labels.pkl"

SAVE_MODEL = True
OUTPUT_PREDICT = True

PREFIX = "10w_"
cmd = 4

def main():
    train_data = joblib.load(TRAIN_DATA)
    train_labels = joblib.load(TRAIN_LABELS)
    test_data = joblib.load(TEST_DATA)
    test_labels = joblib.load(TEST_LABELS)
    print("数据集导入完成")


    if cmd == 1 :
        SAVE_MODEL_NAME = PREFIX + "CPmodel.h5"
        model = myModels.complete_model()
        history = model.fit([train_data['vectors'], train_data['user_actions'], train_data['shop_actions']], train_labels, epochs=EPOCHS,
                            validation_data=([test_data['vectors'], test_data['user_actions'], test_data['shop_actions']], test_labels))
    elif cmd == 2:
        SAVE_MODEL_NAME = PREFIX + "CNN.h5"
        model = myModels.model_CNN()
        history = model.fit(train_data['vectors'], train_labels, epochs=EPOCHS,  validation_data=(test_data['vectors'], test_labels))
    elif cmd == 3:
        SAVE_MODEL_NAME = PREFIX + "origin.h5"
        model = myModels.model_origin()
        history = model.fit([train_data['vectors'], train_data['user_actions'], train_data['shop_actions']], train_labels, epochs=EPOCHS,
                            validation_data=([test_data['vectors'], test_data['user_actions'], test_data['shop_actions']], test_labels))
    elif cmd == 4:
        SAVE_MODEL = False
        SAVE_MODEL_NAME = PREFIX + "CNN5*5.h5"
        model = myModels.model_CNN5()
        history = model.fit(train_data['vectors'], train_labels, epochs=EPOCHS,
                            validation_data=(test_data['vectors'], test_labels))

    if OUTPUT_PREDICT:
        predicts = model.predict([test_data['vectors'], test_data['user_actions'], test_data['shop_actions']])
        print(normalization(predicts))

    plt.title(SAVE_MODEL_NAME)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()


    #serverChan.sendMessage("模型训练完成")
    if SAVE_MODEL:
        model.save(SAVE_MODEL_NAME)
    return 0

if __name__ == "__main__":
    main()