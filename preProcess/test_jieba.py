import jieba
import gensim
import pandas as pd
import serverChan
import numpy as np

def main():
    jieba.enable_paddle()  # 启动paddle模式。 0.40版之后开始支持，早期版本不支持
    s = jieba.lcut("无聊/万圣节弄得好无聊/都没人去的/无聊啊", use_paddle=True)

    print(s)

if __name__ == "__main__":
    main()