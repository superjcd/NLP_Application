import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from settings import categories


data = pd.read_csv('toutiao.csv')

res = data[data.category.isin(categories)]


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42) # 分成一组train, test
for train_index, test_index in split.split(res, res["category"]):  # 基于income_cat进行数据的随机分割
    strat_train_set = res.loc[train_index, :]
    strat_test_set = res.loc[test_index, :]


strat_train_set.dropna(axis='rows').to_csv('news_train.csv', index=False, encoding='utf_8_sig')
strat_test_set.dropna(axis='rows').to_csv('news_test.csv', index=False, encoding='utf_8_sig')

