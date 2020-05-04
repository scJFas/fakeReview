import numpy as np
import pandas as pd
import joblib
from preProcess.data_clear import clear_notNum

INPUT_FILE = "10w_keep_only_chinese.csv"

OUTPUT_FILE= "10w_keep_only_chinese_2label.csv"

OUTPUT_NAME = ""

def divide_2(meta_data):
    labels = []
    if isinstance(meta_data['logreason'][0], str):
        print('cant divide label')
        return
    for i in range(len(meta_data)):
        if(meta_data['logreason'][i] < 0):
            meta_data['logreason'][i] = 0
        else:
            meta_data['logreason'][i] = 1
        labels.append(meta_data['logreason'][i])

    return np.array(labels)

def main():
    meta_data = pd.read_csv(INPUT_FILE, header=0)
    labels = np.array(divide_2(meta_data))
    print("finish convert:", labels.shape)

    # filename = OUTPUT_NAME + ".pkl"
    # joblib.dump(labels, filename)
    # print('finish save:', filename)

    meta_data.to_csv(OUTPUT_FILE, header=True, index=False)

if __name__ == "__main__":
    main()