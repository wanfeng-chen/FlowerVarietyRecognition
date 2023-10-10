import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as transforms

# 定义读取图片的函数
def read_img(root, mode, class_dict):
    if mode == 'train':
        train_path = os.path.join(root, 'train/')
        class_list = os.listdir(train_path)
        img_path_list = []
        label_list =[]
        for i in class_list:
            img_path = os.path.join(train_path, i)
            img_name_class_list = os.listdir(img_path)
            for j in img_name_class_list:
                img_path_list.append(os.path.join(img_path,j))
                label_list.append(class_dict[i])
    elif mode == 'val':
        val_path = os.path.join(root, 'val')
        class_list = os.listdir(val_path)
        img_path_list = []
        label_list = []
        for i in class_list:
            img_path = os.path.join(val_path, i)
            img_name_class_list = os.listdir(img_path)
            for j in img_name_class_list:
                img_path_list.append(os.path.join(img_path, j))
                label_list.append(class_dict[i])
    else:
        test_path = os.path.join(root, 'test')
        class_list = os.listdir(test_path)
        img_path_list = []
        label_list = []
        for i in class_list:
            img_path = os.path.join(test_path, i)
            img_name_class_list = os.listdir(img_path)
            for j in img_name_class_list:
                img_path_list.append(os.path.join(img_path, j))
                label_list.append(class_dict[i])
    return img_path_list, label_list

# 定义图片转换函数
def transform(img ,label, size):
    transform_img = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        ]
    )
    img = transform_img(img)
    label = torch.tensor(label)
    return img, label

# 定义数据集类
class build_dataset(Dataset):
    def __init__(self, root, class_dict, mode, size):
        super(build_dataset, self).__init__()
        self.root = root
        self.class_dict = class_dict
        self.mode = mode
        self.size = size

        self.img_list, self.label_list = read_img(root, mode, class_dict)

    def __getitem__(self, item):
        img = self.img_list[item]
        label = self.label_list[item]
        img = Image.open(img).convert('RGB')
        img ,label = transform(img, label, self.size)

        return img, label
    def __len__(self):
        return len(self.img_list)

if __name__ == "__main__":
    root = './datasets/'
    mode = 'val'
    class_dict = {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}
    size = (32,32)

    # 加载数据集
    Load_train = build_dataset(root,class_dict,mode,size)

    # 定义数据加载器
    from torch.utils.data import DataLoader
    train_data = DataLoader(Load_train, batch_size=1, shuffle=False, num_workers=0)

    # 遍历数据集
    for img ,label in train_data:
        img_data = img
        img_label = label
        print(img_data)
        print(img_label)


