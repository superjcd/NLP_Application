import os
import multiprocessing

# 路径
CUR_LOC = os.path.dirname(os.path.abspath(__file__))
BASE_LOC = '/'.join(CUR_LOC.split('/')[:-1])
CORPUS_LOC = BASE_LOC + '/corpus/wiki/'
MODEL_LOC = 'model'

# 语言， 会体现在模型的名称上
Lan = 'zh'

# 分词工具，可选'pku'（北大分词）， 'thu'（清华分词）， 'jieba'(结巴分词), 'hanlp'(hanlp)
WORD_SEG_TOOL = 'pku'

# 文本压缩文件， 下载地址：https://dumps.wikimedia.org/zhwiki/
CORPUS_BZ = CORPUS_LOC + 'zhwiki-20190420-pages-articles-multistream.xml.bz2'

# 文本解压文件
CORPUS_RAW = CORPUS_LOC + 'wiki_raw.txt'

# 繁体转简体后的文件地址
CORPUS_SIM = CORPUS_LOC + 'wiki_sim.txt'

# 分词后的地址
CORPUS_SEG = CORPUS_LOC + 'wiki_segment.txt'

# 分词文件的匹配模式(本程序会使用正则表达式会匹配CORPUS_LOC目录下的目标分词文件)
SEGFILE_PATTERN = 'segment.txt'

# 词向量训练方式
STRATEGY = 'word2vec'  # 可选'word2vec'(Word2Vec) 或 'fasttext'

# Word2Vec 模型训练参数
# 具体参数及意义见：https://radimrehurek.com/gensim/models/word2vec.html
W2VecParams ={
    'window':5,
    'size':300, # 词向量维度
    'sg':0, # 使用CBOW
    'hs':0, # 使用负采样
    'workers':multiprocessing.cpu_count() # cpu核数量
}


# FastText
FasttextParams = {
    'window':5,
    'size':300, # 词向量维度
    'sg':0, # 使用CBOW
    'hs':0, # 使用负采样
    'workers':multiprocessing.cpu_count() # cpu核数量
}

