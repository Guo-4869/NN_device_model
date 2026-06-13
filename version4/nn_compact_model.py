# nn_compact_model.py
"""
Neural Network Compact Model for MOSFET
Based on: "A SPICE-compatible Neural Network Compact Model for Efficient IC Simulations"
Tung et al., 2024

Key features:
- 2 hidden layers (10-50 neurons each)
- ISRU activation function: x / sqrt(1 + x^2)
- Output transformation: ID = VDS * exp(y1)
- Loss includes derivatives (gm, gds)
"""

import torch
import torch.nn as nn
import numpy as np


class ISRU(nn.Module):
    """ISRU (Inverse Square Root Unit) Activation Function
    ISRU(x) = x / sqrt(1 + x^2)
    
    From the paper: ISRU performs best due to its simpler form 
    and lack of exponential function
    """
    def __init__(self):
        super(ISRU, self).__init__()
    
    def forward(self, x):
        return x / torch.sqrt(1 + x * x + 1e-8)


class BSIM_NN_IV(nn.Module):
    """
    BSIM-NN IV特性神经网络模型
    结构: 输入层 -> 隐藏层1 -> 隐藏层2 -> 输出层
    输出: y1 = ln(ID/VDS)
    
    根据论文: ID = VDS * exp(y1)
    """
    
    def __init__(self, 
                 input_dim: int = 3,      # VGS, VDS, VBS (或包含L, W)
                 hidden_size: int = 10,   # 隐藏层神经元数量
                 num_hidden_layers: int = 2,
                 activation: str = 'isru'):
        """
        参数:
            input_dim: 输入维度 (VGS, VDS, VBS, L, W, Temp等)
            hidden_size: 每层隐藏层神经元数量
            num_hidden_layers: 隐藏层数量
            activation: 激活函数 ('isru', 'tanh', 'sigmoid')
        """
        super(BSIM_NN_IV, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        
        # 选择激活函数
        if activation == 'isru':
            self.activation = ISRU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU()
        
        # 构建网络层
        layers = []
        
        # 输入层 -> 第一隐藏层
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(self.activation)
        
        # 额外的隐藏层
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(self.activation)
        
        # 输出层 (y1)
        layers.append(nn.Linear(hidden_size, 1))
        
        self.network = nn.Sequential(*layers)
        
        # 输入归一化参数 (将从训练数据中计算)
        self.register_buffer('input_means', torch.zeros(input_dim))
        self.register_buffer('input_stds', torch.ones(input_dim))
        
        # 输出归一化参数
        self.register_buffer('output_mean', torch.zeros(1))
        self.register_buffer('output_std', torch.ones(1))
        
    def forward(self, x: torch.Tensor, compute_derivatives: bool = True):
        """
        前向传播
        
        输入: x = [VGS, VDS, VBS, ...]
        输出: ID = VDS * exp(y1)
        """
        # 归一化输入
        x_norm = (x - self.input_means) / (self.input_stds + 1e-8)
        
        # 神经网络前向传播
        y1 = self.network(x_norm).squeeze()
        
        # 反归一化输出 (如果需要)
        y1 = y1 * self.output_std + self.output_mean
        
        # 计算ID (论文公式1: ID = VDS * exp(y1))
        # 添加数值稳定性
        y1_clipped = torch.clamp(y1, -20, 20)
        VDS = x[:, 1]
        ID = VDS * torch.exp(y1_clipped)
        
        # 确保ID在合理范围内
        ID = torch.clamp(ID, min=1e-15, max=1e-2)
        
        result = {
            'ID': ID,
            'y1': y1,
            'VGS': x[:, 0],
            'VDS': VDS,
            'VBS': x[:, 2] if x.shape[1] > 2 else torch.zeros_like(VDS)
        }
        
        # 计算导数 (gm, gds) 用于损失函数
        if compute_derivatives:
            gm, gds = self._compute_derivatives(x)
            result['gm'] = gm
            result['gds'] = gds
        
        return result
    
    def _compute_derivatives(self, x):
        """
        使用有限差分法计算gm和gds
        这种方法更稳定，避免计算图重复问题
        """
        VGS = x[:, 0].detach()
        VDS = x[:, 1].detach()
        VBS = x[:, 2] if x.shape[1] > 2 else torch.zeros_like(VGS)
        
        eps = 1e-4  # 微小扰动
        
        gm_list = []
        gds_list = []
        
        # 批处理计算
        batch_size = x.shape[0]
        
        for i in range(batch_size):
            # 计算gm (对VGS求导)
            vgs_nom = VGS[i].clone()
            vds_nom = VDS[i].clone()
            vbs_nom = VBS[i].clone() if x.shape[1] > 2 else torch.tensor(0.0)
            
            # 前向传播函数
            def get_id(vgs, vds, vbs):
                x_local = torch.stack([vgs, vds, vbs]).unsqueeze(0)
                x_norm = (x_local - self.input_means) / (self.input_stds + 1e-8)
                y1_local = self.network(x_norm).squeeze()
                y1_local = y1_local * self.output_std + self.output_mean
                id_local = vds * torch.exp(torch.clamp(y1_local, -20, 20))
                return id_local
            
            # 中心差分计算gm
            id_plus = get_id(vgs_nom + eps, vds_nom, vbs_nom)
            id_minus = get_id(vgs_nom - eps, vds_nom, vbs_nom)
            gm_i = (id_plus - id_minus) / (2 * eps)
            
            # 中心差分计算gds
            id_plus = get_id(vgs_nom, vds_nom + eps, vbs_nom)
            id_minus = get_id(vgs_nom, vds_nom - eps, vbs_nom)
            gds_i = (id_plus - id_minus) / (2 * eps)
            
            gm_list.append(gm_i)
            gds_list.append(gds_i)
        
        gm = torch.stack(gm_list).squeeze()
        gds = torch.stack(gds_list).squeeze()
        
        # 限制范围
        gm = torch.clamp(gm, min=-1e-2, max=1e-2)
        gds = torch.clamp(gds, min=-1e-3, max=1e-3)
        
        return gm, gds
    
    def set_normalization_params(self, X_train, y_train):
        """设置归一化参数"""
        self.input_means = torch.tensor(X_train.mean(axis=0), dtype=torch.float32)
        self.input_stds = torch.tensor(X_train.std(axis=0) + 1e-8, dtype=torch.float32)
        self.output_mean = torch.tensor(y_train.mean(), dtype=torch.float32)
        self.output_std = torch.tensor(y_train.std() + 1e-8, dtype=torch.float32)


class NNLoss(nn.Module):
    """
    神经网络紧凑模型损失函数 (基于论文公式3)
    
    loss = a*RMS(y1) + b*RMS(gm) + c*RMS(gds)
    """
    
    def __init__(self, 
                 weight_y1: float = 1.0,
                 weight_gm: float = 0.1,
                 weight_gds: float = 0.1):
        super(NNLoss, self).__init__()
        self.weight_y1 = weight_y1
        self.weight_gm = weight_gm
        self.weight_gds = weight_gds
        
        self.mse = nn.MSELoss()
        
    def forward(self, predictions, targets):
        """
        计算损失
        
        参数:
            predictions: 模型输出字典
            targets: 目标值字典 (包含y1, gm, gds)
        """
        loss = 0.0
        
        # y1损失
        if 'y1' in predictions and 'y1' in targets:
            loss += self.weight_y1 * self.mse(predictions['y1'], targets['y1'])
        
        # gm损失 (跨导)
        if 'gm' in predictions and predictions['gm'] is not None:
            if 'gm' in targets and targets['gm'] is not None:
                if not torch.isnan(predictions['gm']).any():
                    loss += self.weight_gm * self.mse(predictions['gm'], targets['gm'])
        
        # gds损失 (输出电导)
        if 'gds' in predictions and predictions['gds'] is not None:
            if 'gds' in targets and targets['gds'] is not None:
                if not torch.isnan(predictions['gds']).any():
                    loss += self.weight_gds * self.mse(predictions['gds'], targets['gds'])
        
        # 检查损失有效性
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(1.0, device=loss.device if torch.is_tensor(loss) else 'cpu',
                               requires_grad=True)
        
        return loss