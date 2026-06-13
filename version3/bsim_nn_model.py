# bsim_nn_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Tuple, Dict

class BSIM_NN_IV(nn.Module):
    """BSIM-NN IV特性神经网络模型"""
    
    def __init__(self, input_dim: int = 5, hidden_dims: list = [16, 16]):
        super(BSIM_NN_IV, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.1))
            if i < len(hidden_dims) - 1:  # 不在最后一层添加BatchNorm
                layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        
        # 输出层: y1, y2p, y2n
        layers.append(nn.Linear(prev_dim, 3))
        
        self.network = nn.Sequential(*layers)
        
        # 可训练参数
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.delta = nn.Parameter(torch.tensor(1e-6))
        self.I0 = nn.Parameter(torch.tensor(1e-12))
        
        # 归一化参数
        self.register_buffer('input_means', torch.tensor([0.0, 0.0, 18.0, 46.0, 0.78]))
        self.register_buffer('input_stds', torch.tensor([0.5, 0.5, 4.0, 6.0, 0.08]))
        
    def forward(self, x: torch.Tensor, compute_derivatives: bool = True) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        关键修复：使用 torch.func 或分离梯度计算
        """
        # 分离输入以便计算梯度 - 重要：需要 requires_grad=True
        VGS = x[:, 0].clone().detach().requires_grad_(compute_derivatives)
        VDS = x[:, 1].clone().detach().requires_grad_(compute_derivatives)
        L = x[:, 2]
        H_FIN = x[:, 3]
        EOT = x[:, 4]
        
        # 归一化输入
        x_norm = self._normalize_inputs(torch.stack([VGS, VDS, L, H_FIN, EOT], dim=1))
        
        # NN输出
        outputs = self.network(x_norm)
        y1, y2p, y2n = outputs[:, 0], outputs[:, 1], outputs[:, 2]
        
        # 限制输出范围
        y1 = torch.clamp(y1, -10, 10)
        y2p = torch.clamp(y2p, -10, 10)
        y2n = torch.clamp(y2n, -10, 10)
        
        # 计算ID
        alpha_val = torch.clamp(self.alpha, 0.1, 1.0)
        tanh_term = torch.tanh(alpha_val * VDS)
        exp_term = torch.exp(y1)
        ID = tanh_term * exp_term
        ID = torch.clamp(ID, min=1e-12, max=1e-2)
        
        # 计算IG
        IG = self._inverse_transform_ig(y2p, y2n)
        IG = torch.clamp(IG, min=-1e-6, max=1e-6)
        
        result = {
            'ID': ID,
            'IG': IG,
            'y1': y1,
            'y2p': y2p,
            'y2n': y2n,
            'VGS': VGS,
            'VDS': VDS
        }
        
        # 计算导数 - 使用不同的计算图避免重复 backward
        if compute_derivatives:
            # 方法1: 分别计算梯度，每次使用 create_graph=True 但只计算一次
            # 计算 gm (dID/dVGS)
            gm = torch.autograd.grad(
                ID.sum(), VGS, 
                create_graph=False,  # 不创建计算图，避免重复
                retain_graph=True,    # 保留计算图供下次使用
                allow_unused=True
            )[0]
            
            # 计算 gds (dID/dVDS)
            gds = torch.autograd.grad(
                ID.sum(), VDS, 
                create_graph=False,
                retain_graph=False,   # 最后一次，释放计算图
                allow_unused=True
            )[0]
            
            # 处理 None 值
            if gm is None:
                gm = torch.zeros_like(VGS)
            if gds is None:
                gds = torch.zeros_like(VDS)
            
            # 限制梯度范围
            gm = torch.clamp(gm, min=-1e-2, max=1e-2)
            gds = torch.clamp(gds, min=-1e-3, max=1e-3)
            
            result['gm'] = gm
            result['gds'] = gds
        
        return result
    
    def _inverse_transform_ig(self, y2p: torch.Tensor, y2n: torch.Tensor) -> torch.Tensor:
        """从y2p, y2n反算IG"""
        I_pos = torch.exp(torch.clamp(y2p, -10, 10)) - self.I0
        I_neg = torch.exp(torch.clamp(y2n, -10, 10)) - self.I0
        return I_pos - I_neg
    
    def _normalize_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """输入归一化"""
        return (x - self.input_means) / (self.input_stds + 1e-8)


class BSIMNNLoss(nn.Module):
    """BSIM-NN损失函数"""
    
    def __init__(self, weights: Dict[str, float] = None):
        super(BSIMNNLoss, self).__init__()
        self.weights = weights or {
            'id': 1.0,
            'gm': 0.05,
            'gds': 0.05,
        }
        self.mse = nn.MSELoss()
        
    def forward(self, predictions: Dict, targets: Dict) -> torch.Tensor:
        loss = 0.0
        valid_loss = True
        
        # ID loss - 使用 log 域
        if 'ID' in predictions and 'ID' in targets:
            pred_log = torch.log(predictions['ID'] + 1e-12)
            target_log = torch.log(targets['ID'] + 1e-12)
            loss += self.weights['id'] * self.mse(pred_log, target_log)
        else:
            valid_loss = False
        
        # gm loss
        if 'gm' in predictions and predictions['gm'] is not None and 'gm' in targets:
            if targets['gm'] is not None and not torch.isnan(targets['gm']).any():
                loss += self.weights['gm'] * self.mse(predictions['gm'], targets['gm'])
        
        # gds loss
        if 'gds' in predictions and predictions['gds'] is not None and 'gds' in targets:
            if targets['gds'] is not None and not torch.isnan(targets['gds']).any():
                loss += self.weights['gds'] * self.mse(predictions['gds'], targets['gds'])
        
        if not valid_loss or torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(1.0, device=loss.device if torch.is_tensor(loss) else 'cpu', 
                               requires_grad=True)
        
        return loss