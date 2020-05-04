import pandas as pd
import numpy as np
import math

INPUT_FILE = "jieba_timeCov_10w.csv"
OUTPUT_FILE = "jieba_timeCov_dataSTA_10w.csv"
IGNORE_TIME = 1e+10

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

def time_discretization(meta_data):
    meta_data.sort_values('updatetime', inplace=True)
    # meta_data.reset_index(inplace=True)
    timestamps = pd.to_datetime(meta_data['updatetime'])
    min_time = timestamps[0]

    deltas = []
    for i in range(len(meta_data)):
        delta = int((timestamps[i] - min_time).to_numpy())
        deltas.append(math.floor(delta/IGNORE_TIME))

    meta_data['updatetime'] = deltas

def main():
    meta_data = pd.read_csv(INPUT_FILE, header=0)
    #time_discretization(meta_data)

    meta_data['userid'] = standardization(meta_data['userid'])
    meta_data['shopid'] = standardization(meta_data['shopid'])
    meta_data['updatetime'] = standardization(meta_data['updatetime'])
    meta_data['star'] = standardization(meta_data['star'])

    meta_data.to_csv(OUTPUT_FILE, header=True, index=False)

if __name__ == "__main__":
    main()