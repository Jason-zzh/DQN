import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FFTNoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(FFTNoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # 基础参数
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        
        # FFT处理参数
        self.eer = nn.Parameter(torch.tensor(4.0))  
        self.errd = nn.Parameter(torch.tensor(3.0))
        self.e2 = nn.Parameter(torch.tensor(7.0))
        self.pr = nn.Parameter(torch.tensor(1.0))
        self.dc = nn.Parameter(torch.tensor(0.0))
        self.cover = nn.Parameter(torch.tensor(1.0))
        self.de_rate = 1.0
        
        self.reset_parameters()
    
    def reset_parameters(self):
        # 初始化权重和偏置
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
        # 初始化FFT参数
        nn.init.constant_(self.eer, 4.0)
        nn.init.constant_(self.errd, 3.0)
        nn.init.constant_(self.e2, 7.0)
        nn.init.constant_(self.pr, 1.0)
        nn.init.constant_(self.dc, 0.0)
        nn.init.constant_(self.cover, 1.0)

    def reset_noise(self):
        # 重置噪声强度参数
        self.pr.data.uniform_(0.9, 1.1)  # 在基础值附近随机扰动
        
        # 可选：轻微扰动其他FFT参数
        self.eer.data += torch.randn_like(self.eer) * 0.1
        self.errd.data += torch.randn_like(self.errd) * 0.1
        self.cover.data += torch.randn_like(self.cover) * 0.05
    
    def _fft_process(self, x):
        # 执行FFT处理
        fft_x = torch.fft.fft2(x)
        
        # 预计算频率网格
        xmax, ymax = fft_x.shape[-2], fft_x.shape[-1]
        x_coords = torch.arange(xmax, device=x.device)
        y_coords = torch.arange(ymax, device=x.device)
        
        # 归一化频率
        mx = 2 * math.pi * x_coords / (xmax - 1)
        my = 2 * math.pi * y_coords / (ymax - 1)
        
        # 向量化计算dx和dy
        mx_sq = mx.unsqueeze(1)**2
        my_sq = my.unsqueeze(0)**2
        dx = (mx.unsqueeze(1)**self.eer) / ((1 + self.errd * mx_sq)**self.e2)
        dy = (my.unsqueeze(0)**self.eer) / ((1 + self.errd * my_sq)**self.e2)
        
        # 计算频域滤波矩阵
        mm = self.pr * (dx + dy)
        
        # 处理DC分量
        mm[0, 0] = self.dc
        
        # 应用滤波
        fd_x = fft_x * (mm + self.cover)
        
        # 衰减pr参数
        self.pr.data *= self.de_rate
        
        # 逆FFT并裁剪
        out = torch.fft.ifft2(fd_x).real
        return torch.clamp(out, torch.min(x), torch.max(x))
    
    def forward(self, input):
        if self.training:
            # 训练时应用FFT处理
            weight = self._fft_process(self.weight)
            bias = self.bias
        else:
            # 测试时使用原始权重
            weight = self.weight
            bias = self.bias
        
        return F.linear(input, weight, bias)
