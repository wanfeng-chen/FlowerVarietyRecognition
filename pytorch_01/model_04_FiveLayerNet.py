import torch  # 导入PyTorch库
import torch.nn as nn  # 导入神经网络模块
import torchvision  # 导入计算机视觉库


class build_FiveLayerNet(nn.Module):
    def __init__(self, num_classes):
        super(build_FiveLayerNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3072, 2042)
        self.fc2 = nn.Linear(2042, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 5)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x


if __name__ == "__main__":
    num_classes = 5  # 定义分类数量为5
    model = build_FiveLayerNet(num_classes)  # 创建LeNet模型实例
    input = torch.randn(2, 3, 32, 32)  # 生成一个随机输入张量，形状为(2, 3, *, *)
    output = model(input)  # 将输入张量传入模型，得到输出
    print(output.shape)  # 打印输出张量的形状


