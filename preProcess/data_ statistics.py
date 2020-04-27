import numpy as np
import pandas as pd

INPUT_FILE = "jieba_timeCov_1w.csv"

def int_status(x) :
    return pd.Series([x.count(),x.min(),x.idxmin(),x.quantile(.25),x.median(),
                      x.quantile(.75),x.mean(),x.max(),x.idxmax(),x.mad(),x.var(),
                      x.std(),x.skew(),x.kurt()],index=['总数','最小值','最小值位置','25%分位数',
                    '中位数','75%分位数','均值','最大值','最大值位数','平均绝对偏差','方差','标准差','偏度','峰度'])
def main():
    meta_data = pd.read_csv(INPUT_FILE, header=0)
    timeDelta = int_status(meta_data['updatetime'])
    user_id = int_status(meta_data['userid'])
    shop_id = int_status(meta_data['shopid'])
    star = int_status(meta_data['star'])

    print("timeDelta: ")
    print(timeDelta)

    print("user_id: ")
    print(user_id)

    print("shop_id: ")
    print(shop_id)

    print("star: ")
    print(star)

if __name__ == "__main__":
    main()