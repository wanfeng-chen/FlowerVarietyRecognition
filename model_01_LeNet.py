import torch  # 导入PyTorch库
import torch.nn as nn  # 导入神经网络模块
import torchvision  # 导入计算机视觉库


class build_LeNet(nn.Module):  # 定义LeNet类，继承自nn.Module
    def __init__(self, num_classes):  # 构造函数，接收分类数量作为参数
        super(build_LeNet, self).__init__()  # 调用父类构造函数
        self.conv1 = nn.Conv2d(3, 32, 5, 1, 2)  # 定义第一个卷积层，输入通道为3，输出通道为32，卷积核大小为5，步长为1，填充为2
        self.maxpool1 = nn.MaxPool2d(2)  # 定义第一个最大池化层，池化核大小为2
        self.conv2 = nn.Conv2d(32, 32, 5, 1, 2)  # 定义第二个卷积层，输入通道为32，输出通道为32，卷积核大小为5，步长为1，填充为2
        self.maxpool2 = nn.MaxPool2d(2)  # 定义第二个最大池化层，池化核大小为2
        self.conv3 = nn.Conv2d(32, 64, 5, 1, 2)  # 定义第三个卷积层，输入通道为32，输出通道为64，卷积核大小为5，步长为1，填充为2
        self.maxpool3 = nn.MaxPool2d(2)  # 定义第三个最大池化层，池化核大小为2
        self.flatten = nn.Flatten()  # 定义展平层，将多维张量展平为一维张量
        self.linear1 = nn.Linear(1024, 64)  # 定义第一个全连接层，输入特征数为1024，输出特征数为64
        self.linear2 = nn.Linear(64, num_classes)  # 定义第二个全连接层，输入特征数为64，输出特征数为分类数量

    def forward(self, x):  # 定义前向传播函数
        x = self.conv1(x)  # 通过第一个卷积层
        x = self.maxpool1(x)  # 通过第一个最大池化层
        x = self.conv2(x)  # 通过第二个卷积层
        x = self.maxpool2(x)  # 通过第二个最大池化层
        x = self.conv3(x)  # 通过第三个卷积层
        x = self.maxpool3(x)  # 通过第三个最大池化层
        x = self.flatten(x)  # 通过展平层
        x = self.linear1(x)  # 通过第一个全连接层
        x = self.linear2(x)  # 通过第二个全连接层
        return x  # 返回输出


if __name__ == "__main__":
    num_classes = 5  # 定义分类数量为5
    model = build_LeNet(num_classes)  # 创建LeNet模型实例
    input = torch.randn(2, 3, 32, 32)  # 生成一个随机输入张量，形状为(2, 3, 32, 32)
    output = model(input)  # 将输入张量传入模型，得到输出
    print(output.shape)  # 打印输出张量的形状
