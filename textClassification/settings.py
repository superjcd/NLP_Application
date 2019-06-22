# 训练数据集
train_data_loc = 'data/news_train.csv'
test_data_loc = 'data/news_test.csv'

# 词向量位置
wordembedding_loc = 'wordemmbedings/zh_wiki.model'

# 需要区分的类别
categories = ['财经', '社会', '科技', '娱乐', '体育', '军事']

# 自动生成类别：{category:i for i,category in zip(list(range(len(categories))), categories) }

categories_class = {'财经': 0, '社会': 1, '科技': 2, '娱乐': 3, '体育': 4, '军事': 5}

#category_lookup = {value:key for key,value in categories_class.items()}
categories_lookup = {0: '财经', 1: '社会', 2: '科技', 3: '娱乐', 4: '体育', 5: '军事'}


model_args = {
    "initial_num_channels":500, # 词向量长度
    "num_classes":6,   # 分类的类别数量
    "num_channels":800 # 模型需要输出的维度（在全脸连接成hi 之前的维度）
}

# trainning params
train_params = {
    'epoches':20,
    'learning_rate':0.001,
    'model_save_directory':'model_storge/'
}

