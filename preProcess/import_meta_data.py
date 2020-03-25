import numpy as np
import pandas as pd

INPUT_FILE = "reviewtext.csv"
OUTPUT_FILE = "output_test.csv"

def import_meta_data():
    meta_data = pd.DataFrame()
    meta_data = pd.read_csv(INPUT_FILE, header=10, names=['reviewid', 'reviewbody', 'logreason', 'updatetime',
                                                          'userid', 'shopid', 'star', 'score1', 'score2', 'score3'])
    return meta_data

def main():
    meta_data = import_meta_data()
    print(meta_data['logreason'])
    #meta_data.to_csv(OUTPUT_FILE, header=True, index= False)

    #print(meta_data['reviewbody'])

if __name__ == "__main__":
    main()



