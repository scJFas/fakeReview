import pandas as pd
import numpy as np
import joblib
import gensim
import math

from preProcess.chinese_word2vec import chinese_word2vec_variable, MODEL_PATH, PART_LENGTH
from preProcess.convert_label import divide_2
import serverChan

TRAIN_PERCENT = 0.9

INPUT_FILE = "jieba_timeCov_1w.csv"

OUTPUT_NAME = "../train/1w"
def convert_user_actions(meta_data):
    actions = []
    if isinstance(meta_data['userid'][0], np.int64) == False or isinstance(meta_data['star'][0], np.int64) == False:
        print('cant convert user actions')
        return
    for i in range(len(meta_data)):
        action = [meta_data['userid'][i], meta_data['star'][i], meta_data['updatetime'][i]]
        actions.append([action])
    return np.array(actions)


def convert_shop_actions(meta_data):
    actions = []
    if isinstance(meta_data['shopid'][0], np.int64) == False or isinstance(meta_data['star'][0], np.int64) == False:
        print('cant convert user actions')
        return
    for i in range(len(meta_data)):
        action = [meta_data['shopid'][i], meta_data['star'][i], meta_data['updatetime'][i]]
        actions.append([action])
    return np.array(actions)

def convert_all(meta_data, model):
    vectors = []
    user_actions = []
    shop_actions = []

    vectors = chinese_word2vec_variable(meta_data, model)
    user_actions = convert_user_actions(meta_data)
    shop_actions = convert_shop_actions(meta_data)

    train_data = {"vectors": vectors, "user_actions": user_actions, "shop_actions":shop_actions}
    return train_data

def main():
    #parts = math.ceil(len(meta_data) / PART_LENGTH)
    parts = 1
    if(parts > 1):
        print("暂时没用")
        #暂时弃用
        #未取得测试集和label
    else:
        model = gensim.models.Word2Vec.load(MODEL_PATH)
        print('model load success')
        meta_data = pd.read_csv(INPUT_FILE, header=0)

        train = meta_data.sample(frac=TRAIN_PERCENT, random_state=0, axis=0)
        test = meta_data[~meta_data.index.isin(train.index)]
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)

        train_data = convert_all(train, model)
        train_labels = divide_2(train)
        test_data = convert_all(test, model)
        test_labels = divide_2(test)

        joblib.dump(train_data, OUTPUT_NAME+"_train_data.pkl")
        joblib.dump(train_labels, OUTPUT_NAME + "_train_labels.pkl")
        joblib.dump(test_data, OUTPUT_NAME + "_test_data.pkl")
        joblib.dump(test_labels, OUTPUT_NAME + "_test_labels.pkl")
        print("file save succeed")

        #clear memory
        train_data = []
        train_labels = []
        test_data = []
        test_labels = []

        serverChan.sendMessage("创建训练与测试数据完成")

    return 0

if __name__ == "__main__":
    main()