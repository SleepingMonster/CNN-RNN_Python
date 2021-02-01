import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.utils.data as Data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import CNN
import CIFAR10
# import resnet18

EPOCHS = 20
BATCH_SIZE = 50
LR = 0.001

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# 获取运行设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
net = CNN.CNN().to(device)


# 对图像进行变换，把图像转成tensor，且归一化
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),   # 随机剪裁
    transforms.RandomHorizontalFlip(),  # 水平翻转
    transforms.ToTensor(),  # 转为张量，并归一化(/255)
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def load_data(file_name):
    # 定义训练数据集
    train_data = CIFAR10.cifar10(
        root=file_name,
        train=True,
        transform=train_transform
    )
    # len(train_loader)=1000，因为一共有50000数据，而BATCH_SIZE=50，所以每一个batch=50
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    # 定义测试数据集
    test_data = CIFAR10.cifar10(
        root=file_name,
        train=False,
        transform=test_transform
    )
    test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)
    return train_loader, test_loader


def show_image(data_loader, count):    # ?
    for image, label in data_loader:
        print(label)    # batch中各个样本的类别
        img = image[0].numpy().transpose(1,2,0)     # 把channel维度放到最后
        plt.imshow(img)
        plt.show()


def train(train_loader):
    # 定义优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)   # 参数：神经网络参数，学习率，动量因子？
    loss_function = nn.CrossEntropyLoss()   # 分类问题：交叉熵作损失函数
    x_pic = []
    loss_count = []
    accuracy_count = []
    for epoch in range(EPOCHS):

        loss_sum_temp = 0
        loss_sum = 0
        accuracy_sum_temp = 0
        accuracy_sum = 0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()  # 每次优化前要清空梯度
            output = net(x)  # cnn output
            loss = loss_function(output, y)  # 计算损失值. 返回的是一个mini-batch的均值！所以累加之后应该除以步数（batch-size=50是步长）
            loss.backward()  # 反向传播，产生梯度
            optimizer.step()  # 更新参数

            accuracy_temp = (torch.max(output, 1)[1].numpy() == y.numpy()).mean()
            accuracy_sum_temp += accuracy_temp
            accuracy_sum += accuracy_temp
            loss_sum_temp += loss.item()
            loss_sum += loss.item()
            if (i+1) % 200 == 0:
                print(loss_sum_temp)
                print('loss is %s and accuracy is %s, when [epoch,batch]=[%s,%s] ' % (loss_sum_temp/200, accuracy_temp, epoch+1, i + 1))
                loss_sum_temp = 0
                accuracy_sum_temp = 0
        accuracy_count.append(accuracy_sum/(50000/BATCH_SIZE))
        loss_count.append(loss_sum/(50000/BATCH_SIZE))
        x_pic.append(epoch+1)

    plt.figure(1)
    plt.subplot(211)
    plt.plot(x_pic, loss_count)
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Pytorch_CNN_Loss & Accuracy')

    plt.subplot(212)
    plt.plot(x_pic, accuracy_count)
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()


def test(test_loader):
    with torch.no_grad():
        correct_num = 0
        for i, (x,y) in test_loader:
            x, y = x.to(device), y.to(device)
            output = net(x)
            _, pred_label = torch.max

def main():
    train_loader, test_loader = load_data('cifar-10-python\cifar-10-batches-py')
    train(train_loader)


if __name__ == "__main__":
    main()
