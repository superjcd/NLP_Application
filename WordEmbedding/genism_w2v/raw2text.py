from gensim.corpora import WikiCorpus
from tqdm import tqdm
from settings import CORPUS_BZ, CORPUS_RAW


def wikicorpus2text(source, target):
    '''
    将维基语料转化成text文本
    :param source: 原始维基语料压缩文件格式(bz2)地址， 下载地址:https://dumps.wikimedia.org/zhwiki/
    :param target: 文件的目标位置
    :return:
    '''
    wiki = WikiCorpus(source, lemmatize=False, dictionary=[])
    with open(target, 'w') as t:
        i = 1
        for text in tqdm(wiki.get_texts()):
            t.write(' '.join(text) + "\n")
            if (i % 10000 == 0):
                print(f'{i * 10000} is done')
            i += 1


if __name__ == '__main__':
    wikicorpus2text(CORPUS_BZ, CORPUS_RAW)