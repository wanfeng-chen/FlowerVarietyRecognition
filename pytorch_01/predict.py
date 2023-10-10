import torch  # 导入PyTorch库
import torchvision.transforms as transforms  # 导入图像预处理库
from PIL import Image  # 导入图像处理库

from model_01_LeNet import build_LeNet
from model_02_AlexNet import build_AlexNet
from model_03_VGG import build_VGG
from model_04_FiveLayerNet import  build_FiveLayerNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 判断是否有GPU可用，如果有则使用GPU，否则使用CPU


def img_transform(img, size):  # 定义图像预处理函数
    transform_img = transforms.Compose(  # 创建图像预处理流程
        [
            transforms.ToTensor(),  # 将图像转换为张量
            transforms.Resize(size),  # 调整图像大小
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 对图像进行归一化处理

        ]
    )
    img = transform_img(img)  # 对输入的图像进行预处理
    return img  # 返回预处理后的图像


def process_img(path_img, size):  # 定义图像处理函数
    img_rgb = Image.open(path_img).convert('RGB')  # 打开图像并将其转换为RGB格式

    img_tensor = img_transform(img_rgb, size)  # 对图像进行预处理并转换为张量
    img_tensor = torch.unsqueeze(img_tensor, dim=0)  # 在第0维度上增加一个维度，将chw格式的张量转换为bchw格式
    img_tensor = img_tensor.to(device)  # 将张量移动到指定的设备（GPU或CPU）上
    return img_tensor, img_rgb  # 返回处理后的张量和原始RGB图像


if __name__ == "__main__":
    path_img = 'R-C.jfif'  # 定义图像路径

    class_dict = {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}  # 定义类别字典
    class_dict_v = {0: 'daisy', 1: 'dandelion', 2: 'roses', 3: 'sunflowers', 4: 'tulips'}  # 定义类别字典的反向映射
    num_classes = 5  # 定义类别数量
    size = (32, 32)  # 定义图像大小

    img_tensor, img_rgb = process_img(path_img, size)  # 处理图像并获取处理后的张量和原始RGB图像
    model = build_FiveLayerNet(num_classes)  # 构建LeNet模型
    model = torch.load("./results/FiveLayerNet_best_weight.pth")  # 加载预训练权重
    with torch.no_grad():  # 不计算梯度
        outputs = model(img_tensor)  # 使用模型对图像张量进行预测
    _, pred_int = torch.max(outputs.data, 1)  # 获取预测结果的最大值索引
    img = Image.open(path_img)  # 打开图像
    img.show()  # 显示图像
    print('图像的预测结果为：{}'.format(class_dict_v[pred_int.item()]))  # 输出预测结果
