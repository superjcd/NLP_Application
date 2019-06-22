import numpy as np
import pandas as pd
import jieba
import torch
from torch.utils.data import  Dataset, DataLoader
from gensim.models import Word2Vec
from settings import categories_class, wordembedding_loc
from model import NewsClassifier

'''
title 数据
# 分词
# 限制clip 长度  
# 读取每一个词
# 
'''
# load wordwmmbeding model
model = Word2Vec.load(wordembedding_loc)
# print(model.wv['中国'])

class NewsData(Dataset):
    def __init__(self, data, num_words, word_emmbeding_dim):
        '''

        :param data: pandas data
        :param num_words:  允许进入网络的字数（从left 开始）
        :param word_emmbeding_dim:  词向量维度
        '''
        self.data = data
        self._num_words = num_words
        self._word_emmbeding_dim = word_emmbeding_dim
        self.size = len(self.data)

    @classmethod
    def from_csv(cls, location, num_words, word_emmbeding_dim):
        data = pd.read_csv(location, dtype=str)
        # 确定'title' 和category
        return cls(data, num_words, word_emmbeding_dim)

    def vectorize(self, sentence, num_words, word_emmbeding_dim=500):
        #sentence = sentence.strip()
        words = list(jieba.cut(sentence))[:num_words]
        words_len = len(words)
        res = []
        for word in words:
            try:
                word_vc = model.wv[word]
            except KeyError:
                word_vc = np.zeros(word_emmbeding_dim)
            res.append(word_vc)
        if words_len < num_words:
            for less_part in range(num_words-words_len):
                res.append(np.zeros(word_emmbeding_dim))
        data = np.array(res)
        return data.T


    def __getitem__(self, index):
        row = self.data.iloc[index]
        X = self.vectorize(row.title, self._num_words, self._word_emmbeding_dim)
        y = categories_class[row.category]
        # 确保返回的数类型正确
        return torch.from_numpy(X).double(), y

    def __len__(self):
        return self.size


if __name__ == '__main__':
    classifer = NewsClassifier(500, 6, 800).float()
    newsdata = NewsData.from_csv('data/news_train.csv', 12, 500)
    dataloader = DataLoader(newsdata, batch_size=10, shuffle=True, drop_last=True)
    for X,y in dataloader:
        #print(X.dtype)
        y_pred = classifer(X.float())