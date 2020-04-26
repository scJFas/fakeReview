import pandas as pd
import numpy as np
import joblib
import gensim
import math

from preProcess.chinese_word2vec import chinese_word2vec_variable, MODEL_PATH, PART_LENGTH
from preProcess.convert_label import divide_2
import serverChan

TRAIN_PERCENT = 0.9

INPUT_FILE = "jieba_chinese_10w.csv"

OUTPUT_NAME = "../train/10w"
def convert_user_actions(meta_data):
    actions = []
    if isinstance(meta_data['userid'][0], np.int64) == False or isinstance(meta_data['star'][0], np.int64) == False:
        print('cant convert user actions')
        return
    return


def convert_shop_actions(meta_data):
    if isinstance(meta_data['shopid'][0], np.int64) == False or isinstance(meta_data['star'][0], np.int64) == False:
        print('cant convert user actions')
        return
    return

def convert_all(meta_data, model):
    vectors = []
    user_actions = []
    shop_actions = []

    vectors = chinese_word2vec_variable(meta_data, model)
    user_actions = convert_user_actions()
    shop_actions = convert_shop_actions()

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
        meta_data = pd.read_csv(INPUT_FILE, header=0)
        model = gensim.models.Word2Vec.load(MODEL_PATH)
        print('model load success')

        train = meta_data.sample(frac=TRAIN_PERCENT, random_state=0, axis=0)
        test = meta_data[~meta_data.index.isin(train.index)]
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)

        train_vectors = chinese_word2vec_variable(train, model)
        train_actions =
        train_labels = divide_2(train)
        test_vectors = chinese_word2vec_variable(test, model)
        test_labels = divide_2(test)

        joblib.dump(train_vectors, OUTPUT_NAME+"_train_vectors.pkl")
        joblib.dump(train_labels, OUTPUT_NAME + "_train_labels.pkl")
        joblib.dump(test_vectors, OUTPUT_NAME + "_test_vector.pkl")
        joblib.dump(test_labels, OUTPUT_NAME + "_test_labels.pkl")
        print("file save succeed")

        serverChan.sendMessage("创建训练与测试数据完成")

    return 0

if __name__ == "__main__":
    main()