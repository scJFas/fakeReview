import pandas as pd
import numpy as np
from sklearn.externals import joblib
import gensim
import math

from preProcess.chinese_word2vec import chinese_word2vec_variable, MODEL_PATH, PART_LENGTH
from preProcess.convert_label import divide_2
import serverChan

TRAIN_PERCENT = 0.9

INPUT_FILE = "jieba_chinese_2columns_smallSize.csv"

OUTPUT_NAME = "mac_data"

def main():
    meta_data = pd.read_csv(INPUT_FILE, header=0)
    model = gensim.models.Word2Vec.load(MODEL_PATH)
    print('model load success')

    parts = math.ceil(len(meta_data) / PART_LENGTH)
    if(parts > 1):
        print("暂时没用")
        #暂时弃用
        #未取得测试集和label
    else:
        end = len(meta_data) - math.floor(len(meta_data) * TEST_PERCENT / 100)
        train = meta_data.sample(frac=TRAIN_PERCENT, random_state=0, axis=0)
        test = meta_data[~meta_data.index.isin(train.index)]
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)

        train_vectors = chinese_word2vec_variable(train, model)
        train_labels = divide_2(train)
        test_vectors = chinese_word2vec_variable(test, model)
        test_labels = divide_2(test)

        joblib.dump(train_vectors, OUTPUT_NAME+"_train_vectors.pkl")
        joblib.dump(train_labels, OUTPUT_NAME + "_train_labels.pkl")
        joblib.dump(test_vectors, OUTPUT_NAME + "_test_vector.pkl")
        joblib.dump(test_labels, OUTPUT_NAME + "_test_labels.pkl")
        print("file save succeed")

        serverChan.sendMessage("创建训练与测试数据完成")


if __name__ == "__main__":
    main()