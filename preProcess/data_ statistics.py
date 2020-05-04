import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = "jieba_timeCov_10w.csv"

def int_status(x):
    return pd.Series([x.count(),x.min(),x.idxmin(),x.quantile(.25),x.median(),
                      x.quantile(.75),x.mean(),x.max(),x.idxmax(),x.mad(),x.var(),
                      x.std(),x.skew(),x.kurt()],index=['总数','最小值','最小值位置','25%分位数',
                    '中位数','75%分位数','均值','最大值','最大值位数','平均绝对偏差','方差','标准差','偏度','峰度'])
def get_review_len(meta_data):
    review_lens = []
    for i in range(len(meta_data)):
        review_lens.append(len(meta_data['reviewbody'][i]))

    meta_data['reviewlen'] = review_lens

def label_count(meta_data):
    label_dict = {}
    for i in range(len(meta_data)):
        num = meta_data["logreason"][i]
        if num not in label_dict:
            label_dict[num] = 1
        else:
            label_dict[num] += 1

    return label_dict

def main():
    meta_data = pd.read_csv(INPUT_FILE, header=0)
    # timeDelta = int_status(meta_data['updatetime'])
    # user_id = int_status(meta_data['userid'])
    # shop_id = int_status(meta_data['shopid'])
    # star = int_status(meta_data['star'])
    #
    # get_review_len(meta_data)
    # lens = int_status(meta_data['reviewlen'])
    #
    # print("timeDelta: ")
    # print(timeDelta)
    #
    # print("user_id: ")
    # print(user_id)
    #
    # print("shop_id: ")
    # print(shop_id)
    #
    # print("star: ")
    # print(star)
    #
    # print("reviewLen: ")
    # print(lens)

    # print("10", meta_data['reviewlen'].quantile(.10))
    # print("90", meta_data['reviewlen'].quantile(.90))

    label_dict = label_count(meta_data)
    print("label_count: ")
    print(label_dict)

    plt.title("label-time")
    plt.xlabel("timeDelta")
    plt.ylabel("label")
    plt.scatter(meta_data['updatetime'], meta_data['logreason'])
    plt.legend('y1')
    plt.show()


if __name__ == "__main__":
    main()