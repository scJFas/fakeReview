import numpy as np
import pandas as pd
import joblib

TRAIN_DATA = "../train/1w_train_data.pkl"
TRAIN_LABELS = "../train/1w_train_labels.pkl"
TEST_DATA = "../train/1w_test_data.pkl"
TEST_LABELS = "../train/1w_test_labels.pkl"

def main():
    train_data = joblib.load(TRAIN_DATA)
    train_labels = joblib.load(TRAIN_LABELS)
    test_data = joblib.load(TEST_DATA)
    test_labels = joblib.load(TEST_LABELS)

    for key in train_data.keys():
        print(f"{key}: {type(train_data[key])} {train_data[key].shape}")

if __name__ == "__main__":
    main()