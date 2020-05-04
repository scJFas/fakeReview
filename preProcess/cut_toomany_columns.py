import pandas as pd

INPUT_FILE = "10w.csv"
OUTPUT_FILE = "10w_cutCol.csv"

def main():
    meta_data = pd.read_csv(INPUT_FILE, header=0)

    #print(meta_data)

    meta_data.to_csv(OUTPUT_FILE, header=True, index=False, columns=['reviewbody', 'logreason', 'updatetime',
                                                          'userid', 'shopid', 'star'])


if __name__ == "__main__":
    main()