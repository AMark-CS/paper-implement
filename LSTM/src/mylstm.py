import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义单个LSTM单元
class My_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(My_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 初始化门的权重和偏置,由于每一个神经元都有自己的偏置，所以在定义单元内部定义
        self.Wf = nn.Parameter(torch.Tensor(input_size + hidden_size, hidden_size))
        self.bf = nn.Parameter(torch.Tensor(hidden_size))
        self.Wi = nn.Parameter(torch.Tensor(input_size + hidden_size, hidden_size))
        self.bi = nn.Parameter(torch.Tensor(hidden_size))
        self.Wo = nn.Parameter(torch.Tensor(input_size + hidden_size, hidden_size))
        self.bo = nn.Parameter(torch.Tensor(hidden_size))
        self.Wg = nn.Parameter(torch.Tensor(input_size + hidden_size, hidden_size))
        self.bg = nn.Parameter(torch.Tensor(hidden_size))
        # 初始化输出层的权重和偏置
        self.W = nn.Parameter(torch.Tensor(hidden_size, output_size))
        self.b = nn.Parameter(torch.Tensor(output_size))
        
    # 用于计算每一种权重的函数
    def cal_weight(self, input, weight, bias):
        return F.linear(input, weight, bias)
    # x是输入的数据,数据的格式是(batch, seq_len, input_size)，包含的是batch个序列，每个序列有seq_len个时间步，每个时间步有input_size个特征
    def forward(self, x):
        # 初始化隐藏层和细胞状态
        h = torch.zeros(1, 1, self.hidden_size).to(x.device)
        c = torch.zeros(1, 1, self.hidden_size).to(x.device)
        # 遍历每一个时间步
        for i in range(x.size(1)):
            input = x[:, i, :].view(1, 1, -1) # 取出每一个时间步的数据
            # 计算每一个门的权重
            f = torch.sigmoid(self.cal_weight(input, self.Wf, self.bf)) # 遗忘门
            i = torch.sigmoid(self.cal_weight(input, self.Wi, self.bi)) # 输入门
            o = torch.sigmoid(self.cal_weight(input, self.Wo, self.bo)) # 输出门
            C_ = torch.tanh(self.cal_weight(input, self.Wg, self.bg)) # 候选值
            # 更新细胞状态
            c = f * c + i * C_
            # 更新隐藏层
            h = o * torch.tanh(c) # 将输出标准化到-1到1之间
        output = self.cal_weight(h, self.W, self.b) # 计算输出
        return output

class My_LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(My_LSTMNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = My_LSTM(input_size, hidden_size)  # 使用自定义的LSTM单元
        self.fc = nn.Linear(hidden_size, output_size)  # 定义全连接层

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))  # LSTM层的前向传播
        out = self.fc(out[:, -1, :])  # 全连接层的前向传播
        return out

