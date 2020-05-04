import numpy as np
import pandas as pd

INPUT_FILE = "10w.csv"
OUTPUT_FILE = "10w.csv"

def import_meta_data():
    meta_data = pd.DataFrame()
    meta_data = pd.read_csv(INPUT_FILE, header=None, names=['reviewid', 'reviewbody', 'logreason', 'updatetime',
                                                          'userid', 'shopid', 'star', 'score1', 'score2', 'score3'])
    return meta_data

def main():
    meta_data = import_meta_data()
    #print(meta_data['logreason'])

    meta_data = meta_data.drop_duplicates()
    meta_data = meta_data.dropna()

    meta_data.to_csv(OUTPUT_FILE, header=True, index= False)

    #print(meta_data['reviewbody'])

if __name__ == "__main__":
    main()



