import torch
import torch.nn as nn
import torch.nn.functional as F



class NewsClassifier(nn.Module):
    def __init__(self, initial_num_channels, num_classes, num_channels):
        '''
        :param initial_num_channels:  原始的通道数， 这里相当于词向量的维度
        :param num_classes:  类别数量
        :param num_channels:  输出的通道数
        '''
        super(NewsClassifier, self).__init__()
        self.convnet = nn.Sequential(
            # (12-kernel_size)/s+1
            nn.Conv1d(in_channels=initial_num_channels,
                      out_channels=num_channels, kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels,
                      kernel_size=3),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels,
                      kernel_size=3),
            nn.ELU()
        )

        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x_surname, apply_softmax=False):
        features_0 = self.convnet(x_surname)
        features = features_0.squeeze(dim=2)
        prediction_vector = self.fc(features)
        if apply_softmax:
            prediction_vector = F.softmax(prediction_vector, dim=1)
        return prediction_vector


if __name__ == '__main__':
    input_data = torch.rand(500, 12).unsqueeze(0)
    print(input_data.shape)
    classifer = NewsClassifier(500, 6, 800)
    y_ = classifer(input_data)
    print(y_)