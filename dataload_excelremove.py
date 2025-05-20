import gc
import nibabel
from torch.utils.data import Dataset, DataLoader
import os
import torch
from config import get_args
import numpy as np
import nibabel
import pandas as pd


# 定义一个自定义数据集类，继承自PyTorch的Dataset类
class DataSet(Dataset):
    def __init__(self, root_path, dir):
        # 使用指定的根路径、目录和Excel文件路径初始化数据集
        self.root_path = root_path
        self.dir = dir
        self.image_path = os.path.join(self.root_path, dir)
        self.images = os.listdir(self.image_path)  # 获取指定目录中的图像文件名列表
        #self.scores_df = pd.read_excel(excel_file)  # 从提供的Excel文件中读取分数数据
    def __getitem__(self, index):
        label = 0  # 初始化标签为0
        image_index = self.images[index]  # 获取给定索引处的图像文件名
        img_path = os.path.join(self.image_path, image_index)  # 创建图像文件的完整路径
        img = nibabel.load(img_path).get_fdata().astype('float32')  # 使用nibabel加载图像数据
        normalization = 'minmax'  # 选择归一化方法（minmax或median）

        # 根据选择的方法对图像数据进行归一化
        if normalization == 'minmax':
            img_max = img.max()
            img = img / img_max
        elif normalization == 'median':
            img_fla = np.array(img).flatten()
            index = np.argwhere(img_fla == 0)
            img_median = np.median(np.delete(img_fla, index))
            img = img / img_median
        img = np.expand_dims(img, axis=0)

        # 根据目录设置标签值
        if self.dir == 'AD/':
            label = 1
        elif self.dir == 'CN/':
            label = 0
        elif self.dir == 'PMCI/':
            label = 2
        elif self.dir == 'SMCI/':
            label = 2

        # 从图像文件名中提取图像ID，并获取相应行的分数数据
        #image_id = image_index.split('.')[0]
        # 假设 image_index 是文件名

        # 清理变量以节省内存
        if normalization == 'minmax':
            del img_max
        else:
            del img_fla, index, img_median
        gc.collect()

        return img,label

    def __len__(self):
        return len(self.images)

# 加载数据的函数
"""def load_data(args, root_path, path1, path2, excel_file):
    train_AD = DataSet(root_path, path1, excel_file)
    train_CN = DataSet(root_path, path2, excel_file)
    train_dataset = train_AD + train_CN
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    del train_dataset
    gc.collect()
    return train_loader"""

def load_data_remove(args, root_path, path1, path2):
    train_AD = DataSet(root_path, path1)
    train_CN = DataSet(root_path, path2)
    #train_PMCI = DataSet(root_path, path3, excel_file)
    #train_SMCI = DataSet(root_path, path4, excel_file)
    #trainDataset = train_AD + train_CN+train_PMCI+ train_SMCI
    trainDataset = train_AD + train_CN
    train_loader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=True)
    del trainDataset
    gc.collect()
    return train_loader

"""# 示例用法
args = {"batch_size": 32}
root_path = 'E:/anten3/mri_example/brain'
path1 = 'AD/'
path2 = 'CN/'
excel_file = 'D:/EDGE download/ADCN_info_ex.xlsx'

train_loader = load_data(args, root_path, path1, path2, excel_file)"""


"""class DataSet(Dataset):
    def __init__(self, root_path, dir):
        self.root_path = root_path
        self.dir = dir
        self.image_path = os.path.join(self.root_path, dir)
        self.images = os.listdir(self.image_path)  # 把路径下的所有文件放在一个列表中

    def __getitem__(self, index):
        label = 0
        image_index = self.images[index]  # 根据索引获取数据名称
        img_path = os.path.join(self.image_path, image_index)  # 获取数据的路径或目录
        img = nibabel.load(img_path).get_fdata()  # 读取数据
        img = img.astype('float32')
        normalization = 'minmax'
        if normalization == 'minmax':
            # 最小最大归一化
            img_max = img.max()
            img = img / img_max
        elif normalization == 'median':
            # 除中位数
            img_fla = np.array(img).flatten()
            index = np.argwhere(img_fla == 0)
            img_median = np.median(np.delete(img_fla, index))
            img = img / img_median

        img = np.expand_dims(img, axis=0)
        # 根据目录名称获取图像标签（AD或CN）
        if self.dir == 'AD/':
            label = label+1
        elif self.dir == 'CN/':
            label = label
        elif self.dir == 'MCI/':
            label = label+2
        if normalization == 'minmax':
            del img_max
        else:
            del img_fla, index, img_median
        gc.collect()
        return img, label

    def __len__(self):
        return len(self.images)

def load_data(args, root_path, path1, path2):
    train_AD = DataSet(root_path, path1)
    train_CN = DataSet(root_path, path2)
    trainDataset = train_AD + train_CN
    train_loader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=True)
    del trainDataset
    gc.collect()
    return train_loader"""



# args = get_args()
# train_data, test_data = load_data(args)
# for step, (b_x, b_y) in enumerate(train_data):
#     if step > 1:
#         break
#
# print(b_x.shape)
# print(b_y.shape)
# print(b_x)
# print(b_y)