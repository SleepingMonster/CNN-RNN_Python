import torch.nn as nn
import torch.nn.functional as F  # 包含torch.nn库中所有函数


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()  # 调用含super的各个的基类__init__函数
        # 卷积块=卷积层（卷积+relu）+池化层，sequential container表示
        self.conv1 = nn.Sequential(
            # 卷积层：补零就是用来得到边缘信息，所以(W:32-F:5+2P:2)/S+1不一定要是整数
            # input(3,32,32), output(16,32,32)
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2),  # 二维卷积
            nn.BatchNorm2d(num_features=16),    # 卷积之后的归一化操作,使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(2)     # 池化层：output(16,16,16)
        )

        self.conv2 = nn.Sequential(
            # 卷积层：input(16,16,16), output(32,16,16)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=32),    # 参数为channel number(深度)
            nn.ReLU(),
            nn.MaxPool2d(2)     # 池化层：output(32,8,8)
        )
        # 全连接层：做分类(pytorch是不显式定义权重和偏置的）
        self.fc1 = nn.Sequential(
            nn.Linear(32*8*8, 1024),
            nn.ReLU()   # 相当于归一化?
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 64),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)   #?? view相当于resize函数，-1表示大小由其他来推出
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    net = CNN()
    print(net)
