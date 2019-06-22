# 基于CNN的文本分类
  使用卷积神经网络（CNN）， 来实现对文本的分类。
  
## 项目结构
```
├── data                     
│   ├── data_manipulation.py
│   ├── news_test.csv
│   ├── news_train.csv
│   └── toutiao.csv
├── dataset.py
├── evaluate.py
├── model.py
├── model_storage
├── settings.py
├── train.py
├── utilities.py
└── wordemmbedings
```
- settings.py: 配置文件
- dataset: 定义了数据进入模型前的的向量表示形式
- model： 定义CNN模型， 主要是基于Conv1d构建的深度卷积模型
- train： 定义了一个训练流程
- utilities: 帮助函数
- data:放置训练数据集及测试数据集, 其中作为示例我放入例头条的新闻数据--toutiao.csv,
然后data_manipulation.py可以自动生成测试及和训练及数据
- wordemmbedings: 词向量所在位置， 词向量的训练方法可以参考[这里](https://github.com/superjcd/NLPRoadMap/tree/master/WordEmbedding/genism_w2v)，
一般来说， 如果使用的是链接中的[gensim](https://radimrehurek.com/gensim/models/word2vec.html)的方式训练词向量的话， 会生成三个文件：**.model,**.syn1neg.py,**.wv.vectors.npy， 这三个文件需要同时被放置在这里
- model_storage:放置我们训练好的CNN文本分类模型
- evlauate.py: 基于测试数据集， 对模型进行评估

## 使用方式
在setting.py中设置好相应的参数之后， 可以运行一下代码：
```python
python train.py
```
运行结束之后可以使用如下代码进行评估:
```python
python evalute.py
```




