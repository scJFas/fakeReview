import pandas as pd
import jieba
import serverChan

INPUT_FILE = "1w.csv"

OUTPUT_FILE = "jieba_1w.csv"

def chinese_jieba(meta_data, show=False):
    jieba.enable_paddle()
    for i in range(len(meta_data['reviewbody'])):
        s1 = meta_data['reviewbody'][i]
        Jieba = jieba.lcut(s1, use_paddle=True)
        s_array = []
        for j in range(len(Jieba)):
            if Jieba[j].find('/') < 0:
                s_array.append(Jieba[j])
        s2 = '/'.join(s_array)
        meta_data['reviewbody'][i] = s2
        if show == True and i % 100 == 0:
            print(i, "/", len(meta_data))


def main():
    meta_data = pd.read_csv(INPUT_FILE, header=0)
    chinese_jieba(meta_data, show=True)
    #print(type(meta_data['reviewbody'][0]))
    meta_data.to_csv(OUTPUT_FILE, header=True, index=False)

    serverChan.sendMessage(title='中文分词完成')

if __name__ == "__main__":
    main()