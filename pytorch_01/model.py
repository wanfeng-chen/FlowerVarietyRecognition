import torch  # 导入PyTorch库
import torch.nn as nn  # 导入神经网络模块
import torchvision  # 导入计算机视觉库


class build_AlexNet(nn.Module):  # 定义AlexNet类，继承自nn.Module
    def __init__(self, num_classes):  # 初始化方法
        super(build_AlexNet, self).__init__()  # 调用父类的初始化方法
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=1, padding=1)  # 定义第一个卷积层
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 定义最大池化层
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1)  # 定义第二个卷积层
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 定义最大池化层
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)  # 定义第三个卷积层
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)  # 定义第四个卷积层
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)  # 定义第五个卷积层
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 定义最大池化层
        self.flatten = nn.Flatten()  # 定义展平层，将多维张量展平为一维张量
        self.linear1 = nn.Linear(4096, 2048)  # 定义第一个全连接层
        self.linear2 = nn.Linear(2048, 1000)  # 定义第二个全连接层
        self.linear3 = nn.Linear(1000, num_classes)  # 定义第三个全连接层

    def forward(self, x):  # 定义前向传播方法
        x = self.conv1(x)  # 对输入x进行第一次卷积、激活和池化操作
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.maxpool3(x)
        x = self.flatten(x)  # 通过展平层
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x  # 返回输出结果


if __name__ == "__main__":
    num_classes = 5  # 定义分类数量为5
    model = build_AlexNet(num_classes)  # 创建LeNet模型实例
    input = torch.randn(2, 3, 32, 32)  # 生成一个随机输入张量，形状为(2, 3, *, *)
    output = model(input)  # 将输入张量传入模型，得到输出
    print(output.shape)  # 打印输出张量的形状
