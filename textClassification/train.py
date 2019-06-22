import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import NewsClassifier
from dataset import NewsData
from settings import model_args, train_params, train_data_loc
from utilities import compute_accuracy


# 获取数据
newsdata = NewsData.from_csv(train_data_loc, 12, 500)

# 导入模型
classifer = NewsClassifier(**model_args).float()

# 定义loss function
loss_func = nn.CrossEntropyLoss()

# 定义优化器
optimizer = optim.Adam(classifer.parameters(), lr=train_params['learning_rate'])
# 定义衰减策略

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer = optimizer,
    mode = 'min',
    factor = 0.5,
    patience = 1
)


def main():
    try:
        for epoch in tqdm(range(train_params['epoches'])):
            print(f'目前的epoch数为:{epoch}>>>>')
            dataloader = DataLoader(newsdata, batch_size=100, shuffle=True, drop_last=True)
            #
            running_loss = 0.0
            running_acc = 0.0
            classifer.train()

            for batch_index, data in enumerate(dataloader):
                X,y = data
                optimizer.zero_grad()
                y_pred = classifer(X.float())
                loss = loss_func(y_pred, y)
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)
                # 进行反向传播
                # 进行反向传播
                loss.backward()
                optimizer.step()
                # 计算精度
                acc_t = compute_accuracy(y_pred, y)
                running_acc += (acc_t - running_acc) / (batch_index + 1)
                if batch_index//10 == 0:
                    print(f'准确率:{acc_t};平均准确率:{running_acc}')

    except KeyboardInterrupt:
        print('强制结束训练')

    finally:
        print('训练结束， 开存储模型')
        trained_time = format(datetime.now(), '%m%d_%H:%M:%S')
        torch.save(classifer, 'model_storage/newsclassifer{}.model'.format(trained_time))



if __name__ == '__main__':
    main()







