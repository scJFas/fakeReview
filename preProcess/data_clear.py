import pandas as pd
import numpy as np

INPUT_FILE = "jieba_chinese_10w.csv"

OUTPUT_FILE = "jieba_chinese_10w.csv"

LOGREASON = ["5","6","16","17","18","-1"]

def clear_notNum(meta_data):
    if isinstance(meta_data['logreason'][0], np.int64):
        return meta_data
    for i in range(len(meta_data)):
        if meta_data['logreason'][i] not in LOGREASON:
            meta_data = meta_data.drop([i])
    return meta_data.reset_index(drop=True)


def main():
    meta_data = pd.read_csv(INPUT_FILE, header=0)
    meta_data = clear_notNum(meta_data)
    meta_data.to_csv(OUTPUT_FILE, header=True, index= False)

if __name__ == "__main__":
    main()