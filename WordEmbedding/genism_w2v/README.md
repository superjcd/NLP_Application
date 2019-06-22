# 基于gensim的词向量训练
  对接[gensim](https://radimrehurek.com/gensim/)提供的word2vec以及fastext模型训练接口, 以及结合不同的分词工具，进行便捷的词向量训练。
  
## 文件构成及作用
| 模块名  | 作用  | 
|---|---|
| raw2text| 提取压缩文件中的文本， 这里主要是用来处理维基开放下载的[原始语料](https://dumps.wikimedia.org/zhwiki/) |
|  tri2sim |将文本中的繁体转化成简体   | 
|  segmentation | 分词|  
|  train | 训练模型 |
|utilities|分词工具等|
|applications|基于模型， 提供项应的应用接口|
| settings| 配置文件|

## 使用方法
 首先需要在settings配置中配置相关参数， 其中最重要的是*CORPUS_LOC*(语料所在的位置)以及*MODEL_LOC*(模型存储位置)，并确保*CORPUS_LOC*和*MODEL_LOC*路径有效。
然后依次配置*WORD_SEG_TOOL*（分词工具）， 关于分词工具的性能评估， 参见[这里](https://nbviewer.jupyter.org/github/superjcd/NLPRoadMap/blob/master/NoteBooks/NLP%E5%B7%A5%E5%85%B7%E6%80%A7%E8%83%BD%E8%AF%84%E6%B5%8B.ipynb)。
然后选择具体的词向量训练方式：**Word2Vec**或**Fasttext**， 已经对应的参数， 部分重要参数已在settings中说明， 
详细信息见如下：
- [word2vec](https://radimrehurek.com/gensim/models/word2vec.html)
- [fasttext](https://radimrehurek.com/gensim/models/fasttext.html#gensim.models.fasttext.FastText)  

假设原始语料是类似于维基开放的bz2格式， 需要依次执行以下步骤来进行模型训练：  
(1)提取压缩文件中的文本， 这里主要是用来处理维基开放下载的原始语料(bz2格式)
```python
# CORPUS_BZ参数指向原始语料
python raw2text.py
```
这会在CORPUS_LOC下面生成*wiki_raw.txt*文件。

(2) 将文本中的繁体转化成简体
```python
python tri2sim.py
```
这会在CORPUS_LOC下面生成*wiki_sim.txt*文件。

(3)分词
```python
python segmentation.py
``` 
这会在CORPUS_LOC下面生成*wiki_segment.txt*文件。  
(4)训练模型
```python
python train.py
```
训练完成后， 会在*MODEL_LOC*下产生相应的模型。  

多数情况下，如果已经准备好了语料， 可以直接从从第三步开始。

## 模型应用
  对于训练好的模型的使用方式可以详见[这里]()







