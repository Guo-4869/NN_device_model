# working_nn_model.py
"""
可工作的BSIM-NN模型 - 简化但能收敛
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


class SimpleBSIM3Model:
    """简化的BSIM3物理模型 - 用于生成数据"""
    
    def calc_ids(self, vgs, vds, vbs=0):
        # 简化的MOSFET模型
        vth = 0.45 + 0.3 * (np.sqrt(0.8 - vbs) - np.sqrt(0.8))
        vth = max(vth, 0.1)
        
        vgst = max(vgs - vth, 0)
        
        # 线性区
        beta = 1e-3  # 简化
        if vds < vgst:
            ids = beta * (2 * vgst - vds) * vds
        else:
            ids = beta * vgst * vgst
        
        # 亚阈值
        if vgs < vth:
            ids = 1e-6 * np.exp((vgs - vth) / 0.078)
        
        ids = max(ids, 1e-12)
        ids = min(ids, 1e-3)
        
        return ids


class WorkingNN(nn.Module):
    """简化的神经网络 - 确保能收敛"""
    
    def __init__(self):
        super(WorkingNN, self).__init__()
        
        # 3 -> 16 -> 16 -> 1
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.Tanh(),  # 使用Tanh而不是ISRU，更容易收敛
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        return self.net(x).squeeze()


def generate_data(n_points=30):
    """生成训练数据"""
    print("Generating data...")
    model = SimpleBSIM3Model()
    
    vgs_list = np.linspace(0, 1.0, n_points)
    vds_list = np.linspace(0, 1.0, n_points)
    
    X = []
    y = []
    
    for vgs in vgs_list:
        for vds in vds_list:
            ids = model.calc_ids(vgs, vds)
            log_ids = np.log10(ids)
            
            X.append([vgs, vds, 0])
            y.append(log_ids)
    
    X = np.array(X)
    y = np.array(y)
    
    # 归一化
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-6
    X_norm = (X - X_mean) / X_std
    
    print(f"Generated {len(X)} samples")
    print(f"log10(ID) range: {y.min():.2f} to {y.max():.2f}")
    
    return X_norm, y, X_mean, X_std


class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train():
    print("="*60)
    print("Working BSIM-NN Training")
    print("="*60)
    
    # 生成数据
    X_norm, y, X_mean, X_std = generate_data(n_points=35)
    
    # 划分数据集
    n = len(X_norm)
    train_n = int(0.8 * n)
    val_n = int(0.1 * n)
    test_n = n - train_n - val_n
    
    indices = np.random.permutation(n)
    train_idx = indices[:train_n]
    val_idx = indices[train_n:train_n+val_n]
    test_idx = indices[train_n+val_n:]
    
    train_dataset = SimpleDataset(X_norm[train_idx], y[train_idx])
    val_dataset = SimpleDataset(X_norm[val_idx], y[val_idx])
    test_dataset = SimpleDataset(X_norm[test_idx], y[test_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    print(f"Train: {train_n}, Val: {val_n}, Test: {test_n}")
    
    # 创建模型
    model = WorkingNN()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # 训练
    print("\nTraining...")
    train_losses = []
    val_losses = []
    
    for epoch in range(500):
        # 训练
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                pred = model(X_batch)
                val_loss += criterion(pred, y_batch).item()
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
    
    # 评估
    print("\n" + "="*60)
    print("Evaluation")
    print("="*60)
    
    model.eval()
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            pred = model(X_batch)
            test_preds.extend(pred.numpy())
            test_targets.extend(y_batch.numpy())
    
    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)
    
    mae = np.mean(np.abs(test_preds - test_targets))
    rmse = np.sqrt(np.mean((test_preds - test_targets)**2))
    
    print(f"Test Results:")
    print(f"  log10(ID) MAE: {mae:.4f}")
    print(f"  log10(ID) RMSE: {rmse:.4f}")
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 预测vs真实
    axes[0].scatter(test_targets, test_preds, alpha=0.5)
    axes[0].plot([test_targets.min(), test_targets.max()], 
                 [test_targets.min(), test_targets.max()], 'r--')
    axes[0].set_xlabel('True log10(ID)')
    axes[0].set_ylabel('Predicted log10(ID)')
    axes[0].set_title(f'MAE={mae:.3f}')
    axes[0].grid(True)
    
    # 误差分布
    error = test_preds - test_targets
    axes[1].hist(error, bins=30, edgecolor='black')
    axes[1].axvline(0, color='r', linestyle='--')
    axes[1].set_xlabel('Error')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Error Distribution')
    axes[1].grid(True)
    
    # 损失曲线
    axes[2].plot(train_losses, label='Train')
    axes[2].plot(val_losses, label='Validation')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Training History')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('working_results.png', dpi=150)
    plt.show()
    
    # 保存
    torch.save({
        'model_state': model.state_dict(),
        'X_mean': X_mean,
        'X_std': X_std,
    }, 'working_model.pth')
    
    print("\nModel saved to working_model.pth")
    
    return model, X_mean, X_std


def generate_verilog(model_path='working_model.pth', output_file='working_bsim_nn.va'):
    """生成Verilog-A代码"""
    
    print("\n" + "="*60)
    print("Generating Verilog-A")
    print("="*60)
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location='cpu')
    model = WorkingNN()
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    X_mean = checkpoint['X_mean']
    X_std = checkpoint['X_std']
    
    # 提取权重
    w1 = model.net[0].weight.detach().numpy()  # [16, 3]
    b1 = model.net[0].bias.detach().numpy()    # [16]
    w2 = model.net[2].weight.detach().numpy()  # [16, 16]
    b2 = model.net[2].bias.detach().numpy()    # [16]
    w3 = model.net[4].weight.detach().numpy()  # [1, 16]
    b3 = model.net[4].bias.detach().numpy()    # [1]
    
    # 生成Verilog-A
    lines = [
        '// Working BSIM-NN Compact Model',
        '// Network: 3 inputs -> 16 neurons (Tanh) -> 16 neurons (Tanh) -> 1 output',
        '',
        '`include "disciplines.vams"',
        '`include "constants.vams"',
        '',
        'module bsim_nn(d, g, s, b);',
        '    inout d, g, s, b;',
        '    electrical d, g, s, b;',
        '',
        '    parameter real L = 1e-6 from (0:inf);',
        '    parameter real W = 10e-6 from (0:inf);',
        '',
        '    real VGS, VDS, VBS, VGS_n, VDS_n, VBS_n, Ids, log10_Id;',
    ]
    
    # 隐藏层变量
    for i in range(16):
        lines.append(f'    real h1_{i}, h2_{i};')
    
    # 归一化参数
    lines.extend([
        '',
        f'    real mean_vgs = {X_mean[0]:.6f};',
        f'    real std_vgs  = {X_std[0]:.6f};',
        f'    real mean_vds = {X_mean[1]:.6f};',
        f'    real std_vds  = {X_std[1]:.6f};',
        f'    real mean_vbs = {X_mean[2]:.6f};',
        f'    real std_vbs  = {X_std[2]:.6f};',
        '',
        '    analog begin',
        '        VGS = V(g, s);',
        '        VDS = V(d, s);',
        '        VBS = V(b, s);',
        '',
        '        VGS_n = (VGS - mean_vgs) / std_vgs;',
        '        VDS_n = (VDS - mean_vds) / std_vds;',
        '        VBS_n = (VBS - mean_vbs) / std_vbs;',
        '',
        '        // Hidden Layer 1 (Tanh)',
    ])
    
    # 第一层
    for i in range(16):
        expr = f"{b1[i]:.6f}"
        expr += f" + {w1[i,0]:.6f}*VGS_n"
        expr += f" + {w1[i,1]:.6f}*VDS_n"
        expr += f" + {w1[i,2]:.6f}*VBS_n"
        lines.append(f'        h1_{i} = tanh({expr});')
    
    lines.append('')
    lines.append('        // Hidden Layer 2 (Tanh)')
    
    # 第二层
    for i in range(16):
        expr = f"{b2[i]:.6f}"
        for j in range(16):
            expr += f" + {w2[i,j]:.6f}*h1_{j}"
        lines.append(f'        h2_{i} = tanh({expr});')
    
    lines.append('')
    lines.append('        // Output Layer')
    
    # 输出层
    expr = f"{b3[0]:.6f}"
    for j in range(16):
        expr += f" + {w3[0,j]:.6f}*h2_{j}"
    lines.append(f'        log10_Id = {expr};')
    
    lines.extend([
        '',
        '        // Clamp output',
        '        log10_Id = (log10_Id < -12) ? -12 : ((log10_Id > -3) ? -3 : log10_Id);',
        '        Ids = pow(10, log10_Id) * (W / 10e-6);',
        '',
        '        if (Ids < 1e-12) Ids = 1e-12;',
        '        if (Ids > 1e-3) Ids = 1e-3;',
        '        if (VDS == 0) Ids = 0;',
        '',
        '        I(d, s) <+ Ids;',
        '        I(g, s) <+ 0;',
        '        I(b, s) <+ 0;',
        '    end',
        '',
        'endmodule'
    ])
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Verilog-A generated: {output_file}")
    
    # 统计
    with open(output_file, 'r') as f:
        line_count = len(f.readlines())
    print(f"Lines: {line_count}")
    
    return output_file


if __name__ == "__main__":
    # 训练
    model, X_mean, X_std = train()
    
    # 生成Verilog-A
    generate_verilog('working_model.pth', 'working_bsim_nn.va')
    
    print("\n" + "="*60)
    print("Next steps:")
    print("="*60)
    print("1. Compile: openvaf working_bsim_nn.va --ngspice -o bsim_nn.so")
    print("2. Test in NGSPICE")
    print("="*60)