import numpy as np
import pandas as pd
import joblib
from preProcess.data_clear import clear_notNum

INPUT_FILE = "jieba_chinese_2columns_smallSize.csv"

OUTPUT_NAME = "chinese_label"

def divide_2(meta_data):
    labels = []
    meta_data = clear_notNum(meta_data)
    for i in range(len(meta_data)):
        if(meta_data['logreason'][i] < 0):
            meta_data['logreason'][i] = 0
        else:
            meta_data['logreason'][i] = 1
        labels.append(meta_data['logreason'][i])

    return np.array(labels)

def main():
    meta_data = pd.read_csv(INPUT_FILE, header=0)
    labels = np.array(divide_2(meta_data, 0))
    print("finish convert:", labels.shape)

    filename = OUTPUT_NAME + ".pkl"
    joblib.dump(labels, filename)
    print('finish save:', filename)

if __name__ == "__main__":
    main()