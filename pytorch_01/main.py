import torch
import random
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from datasets import build_dataset
from model_01_LeNet import build_LeNet
from model_02_AlexNet import build_AlexNet
from model_03_VGG import build_VGG
from model_04_FiveLayerNet import  build_FiveLayerNet



# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)  # python
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # pytorch
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    # 定义超参数
    class CFG:
        seed = 42
        num_classes = 5
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        root = './datasets/'
        class_dict = {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}
        size = (32, 32)
        BATCH_SIZE = 256
        lr = 0.001
        lr_decay_step = 1
        MAX_EPOCH = 100
        start_epoch = 0


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 构建MyDataset实例
    train_data = build_dataset(root=CFG.root, class_dict=CFG.class_dict, mode="train", size=CFG.size)
    valid_data = build_dataset(root=CFG.root, class_dict=CFG.class_dict, mode="val", size=CFG.size)

    train_data_size = len(train_data)
    valid_data_size = len(valid_data)
    print('train_data_size :{}'.format(train_data_size))
    print('valid_data_size :{}'.format(valid_data_size))

    # 构建DataLoder
    train_loader = DataLoader(dataset=train_data, batch_size=CFG.BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=4)

    # 构建模型
    model = build_LeNet(CFG.num_classes).to(device)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 定义优化器
    optimizer = optim.SGD(model.parameters(), lr=CFG.lr, momentum=0.9)  # 选择优化器

    # 定义学习率下降策略
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=CFG.lr_decay_step, gamma=0.1)

    # 定义训练和验证的损失和正确率
    train_curve = list()
    valid_curve = list()
    best_acc = 0

    # 开始训练
    for epoch in range(CFG.start_epoch + 1, CFG.MAX_EPOCH + 1):
        print('【---------第{}轮训练----------】'.format(epoch))
        loss_mean = 0.
        train_step = 0

        model.train()
        for i, data in enumerate(train_loader):
            # 前向传播
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # 反向传播
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()

            # 更新权重
            optimizer.step()

            train_step += 1
            if train_step % 10 == 0:
                print('训练次数：{}， Loss：{}'.format(train_step, loss.item()))

        total_val_loss = 0
        correct = 0.
        with torch.no_grad():
            print('---------第{}轮测试----------'.format(epoch))
            for i, data in enumerate(valid_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                loss = criterion(outputs, labels)

                total_val_loss += loss.item()
                # 统计分类情况
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).squeeze().cpu().sum().numpy()
            print('测试正确率是：{}'.format(correct / valid_data_size))
            print('mean_loss:{}'.format(total_val_loss / valid_data_size))

        # 保存最好的模型
        if (correct / valid_data_size) > best_acc:
            best_acc = correct / valid_data_size
            torch.save(model, "./results/LeNet_best_weight(BatchSize256).pth")
            print('模型已保存')
