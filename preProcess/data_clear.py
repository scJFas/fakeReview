import pandas as pd
import numpy as np

INPUT_FILE = "reviewtext.csv"

OUTPUT_FILE = "reviewtext_cleared.csv"

LOGREASON = ["5","6","16","17","18","-1"]

def clear_notNum(meta_data):
    if isinstance(meta_data['logreason'][0], np.int64):
        return meta_data
    for i in range(len(meta_data)):
        if meta_data['logreason'][i] not in LOGREASON:
            meta_data = meta_data.drop([i])
    return meta_data.reset_index(drop=True)

def clear_all_dataTypes(meta_data):

    if isinstance(meta_data['reviewid'][0], np.int64) == False:
        for i in range(len(meta_data)):
            if str.isdigit(meta_data['reviewid'][i]) == False:
                meta_data.drop([i], inplace=True)
    meta_data.reset_index(drop=True, inplace=True)

    if isinstance(meta_data['logreason'][0], np.int64) == False:
        for i in range(len(meta_data)):
            if meta_data['logreason'][i] not in LOGREASON:
                meta_data.drop([i], inplace=True)
    meta_data.reset_index(drop=True, inplace=True)

    if isinstance(meta_data['updatetime'][0], str):
        for i in range(len(meta_data)):
            try:
                pd.to_datetime(meta_data['updatetime'][i], format='%Y/%m/%d %H:%M')
            except ValueError:
                meta_data.drop([i], inplace=True)
            except :
                print("pd.to_datetime unknown error")
                return []
    meta_data.reset_index(drop=True, inplace=True)

    if isinstance(meta_data['userid'][0], np.int64) == False:
        for i in range(len(meta_data)):
            if str.isdigit(meta_data['userid'][i]) == False:
                meta_data.drop([i], inplace=True)
    meta_data.reset_index(drop=True, inplace=True)
    if isinstance(meta_data['shopid'][0], np.int64) == False:
        for i in range(len(meta_data)):
            if str.isdigit(meta_data['shopid'][i]) == False:
                meta_data.drop([i], inplace=True)
    meta_data.reset_index(drop=True, inplace=True)
    if isinstance(meta_data['star'][0], np.int64) == False:
        for i in range(len(meta_data)):
            if isinstance(meta_data['star'][i], str):
                if str.isdigit(meta_data['star'][i]) == False:
                    meta_data.drop([i], inplace=True)
    meta_data.reset_index(drop=True, inplace=True)
    # if isinstance(meta_data['score1'][0], np.int64) == False:
    # if isinstance(meta_data['score2'][0], np.int64) == False:
    # if isinstance(meta_data['score3'][0], np.int64) == False:

    return meta_data


def main():
    meta_data = pd.read_csv(INPUT_FILE, header=0)
    print(f"original len: {len(meta_data)}")
    meta_data = clear_all_dataTypes(meta_data)
    print(f"after cleared len: {len(meta_data)}")
    meta_data.to_csv(OUTPUT_FILE, header=True, index= False)

if __name__ == "__main__":
    main()