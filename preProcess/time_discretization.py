import pandas as pd
import numpy as np
import math

INPUT_FILE = "jieba_1w.csv"
OUTPUT_FILE = "jieba_timeCov_1w.csv"
IGNORE_TIME = 1e+10

def time_discretization(meta_data):
    meta_data.sort_values('updatetime', inplace=True)
    meta_data.reset_index(inplace=True)
    timestamps = pd.to_datetime(meta_data['updatetime'])
    min_time = timestamps[0]

    deltas = []
    for i in range(len(meta_data)):
        delta = int((timestamps[i] - min_time).to_numpy())
        deltas.append(math.floor(delta/IGNORE_TIME))

    meta_data['updatetime'] = deltas

def main():
    meta_data = pd.read_csv(INPUT_FILE, header=0)
    time_discretization(meta_data)

    meta_data.to_csv(OUTPUT_FILE, header=True, index=False)

if __name__ == "__main__":
    main()