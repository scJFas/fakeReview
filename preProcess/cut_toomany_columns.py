import pandas as pd

INPUT_FILE = "keep_only_chinese.csv"
OUTPUT_FILE = "2columns.csv"

def main():
    meta_data = pd.read_csv(INPUT_FILE, header=0)

    #print(meta_data)

    meta_data.to_csv(OUTPUT_FILE, index=False, columns=['reviewbody', 'logreason'])


if __name__ == "__main__":
    main()