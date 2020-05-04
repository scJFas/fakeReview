import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import joblib
import matplotlib.pyplot as plt

import serverChan

TEST_DATA = "10w_dataSTA_test_data.pkl"
TEST_LABELS = "10w_dataSTA_test_labels.pkl"

MODEL1 = "10w_CNN.h5"
MODEL2 = "10w_CPmodel.h5"
MODEL3 = "10w_origin.h5"

THRESHOLDS = 10000

AUC = True

def get_single_array(a):
    b = []
    for value in a:
        b.append(value[0])

    return b

def main():

    test_data = joblib.load(TEST_DATA)
    test_labels = joblib.load(TEST_LABELS)
    print("数据集导入完成")

    model = tf.keras.models.load_model(MODEL1)
    pred1 = model.predict(test_data['vectors'])

    model = tf.keras.models.load_model(MODEL2)
    pred2 = model.predict([test_data['vectors'], test_data['user_actions'], test_data['shop_actions']])

    model = tf.keras.models.load_model(MODEL3)
    pred3 = model.predict([test_data['vectors'], test_data['user_actions'], test_data['shop_actions']])

    pred1_array = get_single_array(pred1)
    pred2_array = get_single_array(pred2)
    pred3_array = get_single_array(pred3)

    if AUC:
        auc1 = tf.keras.metrics.AUC(num_thresholds=THRESHOLDS)
        auc2 = tf.keras.metrics.AUC(num_thresholds=THRESHOLDS)
        auc3 = tf.keras.metrics.AUC(num_thresholds=THRESHOLDS)

        auc1.update_state(test_labels, pred1_array)
        auc2.update_state(test_labels, pred2_array)
        auc3.update_state(test_labels, pred3_array)

        print(MODEL1," AUC: ", auc1.result().numpy())
        print(MODEL2," AUC: ", auc2.result().numpy())
        print(MODEL3," AUC: ", auc3.result().numpy())

if __name__ == "__main__":
    main()