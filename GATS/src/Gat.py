import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch  # 张量库
import torch.nn.functional as Func  # 激活函数
import torch.nn as nn  # 神经网络模块
from sklearn.preprocessing import LabelEncoder # 标签编码
from torch.utils.data import DataLoader, TensorDataset # 数据加载器，数据集


from utils import dataprocess

# 加载数据并处理
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = dataprocess.load_data()

