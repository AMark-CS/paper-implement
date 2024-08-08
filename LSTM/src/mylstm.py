import torch
import torch.nn as nn
import torch.nn.functional as F

# ���嵥��LSTM��Ԫ
class My_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(My_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # ��ʼ���ŵ�Ȩ�غ�ƫ��,����ÿһ����Ԫ�����Լ���ƫ�ã������ڶ��嵥Ԫ�ڲ�����
        self.Wf = nn.Parameter(torch.Tensor(input_size + hidden_size, hidden_size))
        self.bf = nn.Parameter(torch.Tensor(hidden_size))
        self.Wi = nn.Parameter(torch.Tensor(input_size + hidden_size, hidden_size))
        self.bi = nn.Parameter(torch.Tensor(hidden_size))
        self.Wo = nn.Parameter(torch.Tensor(input_size + hidden_size, hidden_size))
        self.bo = nn.Parameter(torch.Tensor(hidden_size))
        self.Wg = nn.Parameter(torch.Tensor(input_size + hidden_size, hidden_size))
        self.bg = nn.Parameter(torch.Tensor(hidden_size))
        # ��ʼ��������Ȩ�غ�ƫ��
        self.W = nn.Parameter(torch.Tensor(hidden_size, output_size))
        self.b = nn.Parameter(torch.Tensor(output_size))
        
    # ���ڼ���ÿһ��Ȩ�صĺ���
    def cal_weight(self, input, weight, bias):
        return F.linear(input, weight, bias)
    # x�����������,���ݵĸ�ʽ��(batch, seq_len, input_size)����������batch�����У�ÿ��������seq_len��ʱ�䲽��ÿ��ʱ�䲽��input_size������
    def forward(self, x):
        # ��ʼ�����ز��ϸ��״̬
        h = torch.zeros(1, 1, self.hidden_size).to(x.device)
        c = torch.zeros(1, 1, self.hidden_size).to(x.device)
        # ����ÿһ��ʱ�䲽
        for i in range(x.size(1)):
            input = x[:, i, :].view(1, 1, -1) # ȡ��ÿһ��ʱ�䲽������
            # ����ÿһ���ŵ�Ȩ��
            f = torch.sigmoid(self.cal_weight(input, self.Wf, self.bf)) # ������
            i = torch.sigmoid(self.cal_weight(input, self.Wi, self.bi)) # ������
            o = torch.sigmoid(self.cal_weight(input, self.Wo, self.bo)) # �����
            C_ = torch.tanh(self.cal_weight(input, self.Wg, self.bg)) # ��ѡֵ
            # ����ϸ��״̬
            c = f * c + i * C_
            # �������ز�
            h = o * torch.tanh(c) # �������׼����-1��1֮��
        output = self.cal_weight(h, self.W, self.b) # �������
        return output

class My_LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(My_LSTMNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = My_LSTM(input_size, hidden_size)  # ʹ���Զ����LSTM��Ԫ
        self.fc = nn.Linear(hidden_size, output_size)  # ����ȫ���Ӳ�

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))  # LSTM���ǰ�򴫲�
        out = self.fc(out[:, -1, :])  # ȫ���Ӳ��ǰ�򴫲�
        return out

