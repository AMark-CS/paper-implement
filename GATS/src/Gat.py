import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch  # ������
import torch.nn.functional as Func  # �����
import torch.nn as nn  # ������ģ��
from sklearn.preprocessing import LabelEncoder # ��ǩ����
from torch.utils.data import DataLoader, TensorDataset # ���ݼ����������ݼ�


from utils import dataprocess

# �������ݲ�����
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = dataprocess.load_data()

