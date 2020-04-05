import pandas as pd
import numpy as np
import jieba
import serverChan
import gensim
import math
from sklearn.externals import joblib

INPUT_FILE = "jieba_chinese_2columns.csv"

OUTPUT_NAME = "chinese_vector"

MODEL_PATH = '../baike_26g_news_13g_novel_229g.model'

SENTENCE_LENGTH = 128

PART_LENGTH = 100000

#return np.array
def chinese_word2vec_variable(meta_data, model):
    vectors = []
    for i in range(len(meta_data)):
        zero_array = [0.0 for _ in range (SENTENCE_LENGTH)]
        s_array = []
        s = meta_data['reviewbody'][i]
        if isinstance(s, float):
            s = ""

        word = ""
        for j in range(len(s)):
            if s[j] == '/':
                s_array.append(word)
                word = ''
            else:
                word += s[j]

        if word != '':
            s_array.append(word)

        for j in range(len(s_array)):
            word = s_array[j]
            try:
                vec = model.wv.__getitem__(word)
            except:
                vec = []
            s_array[j] = vec

        while(True):
            try:
                s_array.remove([])
            except ValueError:
                break
            except :
                print("s_array.remove unknown error")
                return []

        #modify to certain length
        while len(s_array) < SENTENCE_LENGTH:
            s_array.append(zero_array)
        if len(s_array) > SENTENCE_LENGTH:
            s_array = s_array[:SENTENCE_LENGTH]

        vectors.append(s_array)

    return np.array(vectors)



def main():
    meta_data = pd.read_csv(INPUT_FILE, header=0)
    model = gensim.models.Word2Vec.load(MODEL_PATH)
    print('model load success')

    parts = math.ceil(len(meta_data) / PART_LENGTH)
    for i in range(parts):
        k = i * PART_LENGTH
        vectors = chinese_word2vec_variable(meta_data[k:k+PART_LENGTH].reset_index(drop=True), model)
        print('finish word2vec:', vectors.shape)
        #print(vectors)
        #np.save(OUTPUT_NP, vectors)
        # 存储过大数据时会报错
        filename = OUTPUT_NAME+f"_{i}.pkl"
        joblib.dump(vectors, filename)
        print('finish save:', filename)
        vectors = []

    serverChan.sendMessage(title='word2vec完成')

if __name__ == "__main__":
    main()