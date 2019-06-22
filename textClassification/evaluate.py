import torch
from dataset import NewsData
from settings import categories_lookup


#data
test_data = NewsData.from_csv('data/news_test.csv', 12, 500)

#model
model = torch.load('model_storage/newsclassifer0527_17:20:56.model').float()
# 这个很重要, 设置成评估模式， 不会进行参数更新
model.eval()

total_nums = 0
acc_nums = 0

for i in range(test_data.size):
    sentence = test_data.data.title[i]
    print(f'句子是：{sentence}')
    category_real = test_data.data.category[i]
    print(f'真实类别是：{category_real}')
    X = test_data.vectorize(sentence, 12, 500)
    y_pred = model(torch.from_numpy(X).unsqueeze(0).float())
    predict_classs = categories_lookup[int(y_pred.argmax())]
    print(f'预测的结果是：{predict_classs}')
    print('*' *10)
    total_nums += 1
    if category_real == predict_classs:
        acc_nums += 1

print(f'final accuracy is {acc_nums/total_nums}')





