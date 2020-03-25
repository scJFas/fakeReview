
import jieba
import gensim
import pandas as pd
import serverChan

MODEL_PATH = 'baike_26g_news_13g_novel_229g.model'

OUTPUT_FILE = 'index2vector.csv'

def main():
    model = gensim.models.Word2Vec.load(MODEL_PATH)
    print('词向量维度：', model.wv.vectors.shape)
    #print(model.wv.vectors)

    index2vector = pd.DataFrame(columns=['index'])
    index2vector['index'] = model.wv.index2word

    index2vector.to_csv(OUTPUT_FILE, header=False, index=False)

    serverChan.sendMessage(title='中文分词测试完毕')


if __name__ == "__main__":
    main()