import numpy as np
import pandas as pd
from sklearn.externals import joblib
INPUT_FILE = "jieba_chinese_2columns_smallSize.csv"

OUTPUT_NAME = "chinese_label"

def divide_2(meta_data):
    labels = []
    for i in range(len(meta_data['logreason'])):
        if(meta_data['logreason'][i] < 0):
            meta_data['logreason'][i] = 0
        else:
            meta_data['logreason'][i] = 1
        labels.append(meta_data['logreason'][i])

    return labels

def main():
    meta_data = pd.read_csv(INPUT_FILE, header=0)
    labels = np.array(divide_2(meta_data))
    print("finish convert:", labels.shape)

    filename = OUTPUT_NAME + ".pkl"
    joblib.dump(labels, filename)
    print('finish save:', filename)

if __name__ == "__main__":
    main()