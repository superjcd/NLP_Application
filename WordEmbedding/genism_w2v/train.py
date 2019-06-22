import multiprocessing
from datetime import datetime
import logging
from gensim.models import Word2Vec, FastText
from utilities import MySentences
from settings import W2VecParams, FasttextParams,Lan, CORPUS_LOC, STRATEGY, MODEL_LOC

# 配置loging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def train_model(source_txt=CORPUS_LOC, method=STRATEGY):
    # source = 'data/wiki_segmented.txt'
    corpus = MySentences(dirname=source_txt)
    # 定义model
    print('INFO：<开始模型训练>')
    if method == 'word2vec':
        model = Word2Vec(corpus, **W2VecParams)
    elif method == 'fasttext':
        model = FastText(corpus, **FasttextParams)
    else:
        raise ValueError('确定你在setting文件中设置的STRATEGY参数是"word2vec"和"fasttext"二者之一吗？')
    print('INFO：<模型训练完成， 开始存储模型>')
    prefix = Lan + '_' + method + format(datetime.now(), '%m%d_%H:%M')
    model.save('{}/{}.model'.format(MODEL_LOC ,prefix))


if __name__ == '__main__':
    # train model
    train_model(method='fasttext')
    


