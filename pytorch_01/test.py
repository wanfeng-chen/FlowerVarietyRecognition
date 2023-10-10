import torch  # 导入PyTorch库
from torch.utils.data import DataLoader  # 导入DataLoader类，用于批量加载数据

from datasets import build_dataset  # 导入自定义的build_dataset函数，用于构建数据集
from model_01_LeNet import build_LeNet
from model_02_AlexNet import build_AlexNet
from model_03_VGG import build_VGG
from model_04_FiveLayerNet import  build_FiveLayerNet

if __name__ =='__main__':  # 当前脚本作为主程序运行时执行以下代码
    class CFG:  # 定义一个配置类，用于存储各种参数
        num_classes = 5  # 类别数量
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 设备选择，如果有GPU则使用GPU，否则使用CPU
        root = './datasets/'  # 数据集根目录
        class_dict = {'daisy':0,'dandelion':1,'roses':2,'sunflowers':3,'tulips':4}  # 类别字典，用于将类别名称映射到整数
        size = (32,32)  # 图像大小
        BATCH_SIZE = 1  # 批量大小

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 再次选择设备，与上面的CFG.device相同

    # 构建MyDataset实例
    test_data = build_dataset(root=CFG.root, class_dict=CFG.class_dict, mode="test", size=CFG.size)  # 使用build_dataset函数构建测试数据集

    test_data_size = len(test_data)  # 计算测试数据集的大小
    print('test_data_size :{}'.format(test_data_size))  # 打印测试数据集的大小

    # 构建DataLoder
    test_loader = DataLoader(dataset=test_data, batch_size=CFG.BATCH_SIZE, shuffle=True)  # 使用DataLoader类构建数据加载器，用于批量加载测试数据

    model = build_LeNet(CFG.num_classes).to(device)  # 使用build_LeNet函数构建LeNet模型，并将模型移动到指定设备上
    model = torch.load("./results/LeNet_best_weight(BatchSize256).pth")  # 加载预训练的模型权重
    correct = 0  # 初始化正确分类的数量为0
    with torch.no_grad():  # 不计算梯度，用于加速计算
        for i, data in enumerate(test_loader):  # 遍历测试数据加载器中的每一个批次
            inputs, labels = data  # 获取输入图像和标签
            inputs, labels = inputs.to(device), labels.to(device)  # 将输入图像和标签移动到指定设备上
            outputs = model(inputs)  # 使用模型对输入图像进行预测，得到输出

            # 统计分类情况
            _, predicted = torch.max(outputs.data, 1)  # 获取预测结果中概率最大的类别

            # print(predicted.item(), labels.item())

            correct += (predicted == labels).squeeze().cpu().sum().numpy()  # 计算预测结果与标签相同的数量，并累加到correct变量中
        print('测试正确率是：{:.4f}'.format(correct / test_data_size))  # 计算并打印测试正确率