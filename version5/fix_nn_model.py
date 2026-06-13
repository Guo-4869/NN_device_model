# fixed_nn_model.py
"""
修复版BSIM-NN模型训练 - 确保收敛
网络结构: 3输入 -> 32神经元 -> 32神经元 -> 1输出
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class ISRU(nn.Module):
    """ISRU激活函数"""
    def forward(self, x):
        return x / torch.sqrt(1 + x * x + 1e-8)


class FixedBSIM_NN(nn.Module):
    """修复版神经网络 - 正确的归一化和输出"""
    
    def __init__(self, input_dim=3, hidden_size=32, num_layers=2):
        super(FixedBSIM_NN, self).__init__()
        
        # 输入归一化层
        self.input_norm = nn.BatchNorm1d(input_dim, affine=False)
        
        # 网络层
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.activation = ISRU()
        
        # Dropout防止过拟合
        self.dropout = nn.Dropout(0.1)
        
        # 初始化
        self._init_weights()
        
    def _init_weights(self):
        for m in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(m.weight, gain=0.5)
            nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        # 归一化
        x_norm = self.input_norm(x)
        
        # 隐藏层
        h1 = self.activation(self.fc1(x_norm))
        h1 = self.dropout(h1)
        h2 = self.activation(self.fc2(h1))
        h2 = self.dropout(h2)
        
        # 输出 log10(ID)
        log10_id = self.fc3(h2).squeeze()
        log10_id = torch.clamp(log10_id, -12, -3)
        
        return log10_id
    
    def set_normalization_stats(self, X_train):
        """设置归一化统计量"""
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-6
        self.input_norm.running_mean = torch.tensor(mean, dtype=torch.float32)
        self.input_norm.running_var = torch.tensor(std**2, dtype=torch.float32)


class AccurateBSIM3Model:
    """精确的BSIM3模型"""
    
    def __init__(self):
        # 工艺参数
        self.tox = 2e-9  # 氧化层厚度
        self.cox = 3.9 * 8.854e-12 / self.tox  # 氧化层电容
        self.vth0 = 0.45  # 零偏阈值电压
        self.u0 = 0.04  # 迁移率
        self.vsat = 8e4  # 饱和速度
        self.vt = 0.026  # 热电压
        
    def calc_ids(self, vgs, vds, vbs=0, l=1e-6, w=10e-6):
        """计算漏极电流"""
        # 阈值电压模型
        gamma = 0.3  # 体效应系数
        phi = 0.8  # 表面势
        vth = self.vth0 + gamma * (np.sqrt(phi - vbs) - np.sqrt(phi))
        vth = np.clip(vth, 0.1, 1.0)
        
        vgst = vgs - vth
        
        # 线性/饱和区
        if vgst > 0:
            # 迁移率退化和速度饱和
            ueff = self.u0 / (1 + 0.3 * vgst)
            esat = 2 * self.vsat / ueff
            vdsat = vgst * esat * l / (vgst + esat * l)
            beta = ueff * self.cox * (w / l)
            
            if vds < vdsat:
                ids = beta * (vgst - vds/2) * vds / (1 + vds/(esat*l))
            else:
                ids = beta * (vgst - vdsat/2) * vdsat / (1 + vdsat/(esat*l))
                ids *= (1 + 0.1 * (vds - vdsat))
        else:
            ids = 0
        
        # 亚阈值电流
        n = 1.2
        ids_sub = 1e-7 * (w/l) * np.exp(vgst / (n * self.vt)) * (1 - np.exp(-vds / self.vt))
        
        ids = ids + ids_sub
        
        # 限制范围
        ids = np.clip(ids, 1e-12, 1e-3)
        
        return ids


def generate_training_data(n_points=60, l=1e-6, w=10e-6):
    """生成高质量训练数据"""
    print("Generating training data...")
    model = AccurateBSIM3Model()
    
    # 更精细的采样
    vgs = np.linspace(0, 1.0, n_points)
    vds = np.linspace(0, 1.0, n_points)
    
    data = []
    for vg in tqdm(vgs, desc="Generating"):
        for vd in vds:
            id_val = model.calc_ids(vg, vd, 0, l, w)
            data.append({
                'VGS': vg, 
                'VDS': vd, 
                'VBS': 0, 
                'ID': id_val,
                'log10_ID': np.log10(max(id_val, 1e-15))
            })
    
    df = pd.DataFrame(data)
    df['log10_ID'] = df['log10_ID'].clip(-12, -3)
    
    print(f"Generated {len(df)} samples")
    print(f"ID range: {df['ID'].min():.2e} - {df['ID'].max():.2e} A")
    print(f"log10(ID) range: {df['log10_ID'].min():.2f} - {df['log10_ID'].max():.2f}")
    
    return df


class MOSFETDataset(Dataset):
    def __init__(self, df):
        self.X = torch.tensor(df[['VGS', 'VDS', 'VBS']].values, dtype=torch.float32)
        self.y = torch.tensor(df['log10_ID'].values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_model(model, train_loader, val_loader, epochs=1000, device='cpu'):
    """修复版训练函数"""
    model = model.to(device)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=100, factor=0.5, verbose=True
    )
    
    # 损失函数
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in tqdm(range(epochs), desc="Training"):
        # 训练
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                val_loss += criterion(pred, y).item()
        
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # 学习率调整
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'fixed_best_model.pth')
        
        # 打印进度
        if (epoch + 1) % 200 == 0:
            print(f"\nEpoch {epoch+1}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}, LR={optimizer.param_groups[0]['lr']:.6f}")
    
    return train_losses, val_losses


def evaluate_model(model, test_loader, device='cpu'):
    """详细评估"""
    model.eval()
    
    all_preds = []
    all_targets = []
    all_X = []
    
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            pred = model(X).cpu()
            all_preds.extend(pred.numpy())
            all_targets.extend(y.numpy())
            all_X.extend(X.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_X = np.array(all_X)
    
    # 计算误差
    mae = np.mean(np.abs(all_preds - all_targets))
    rmse = np.sqrt(np.mean((all_preds - all_targets)**2))
    max_error = np.max(np.abs(all_preds - all_targets))
    
    print(f"\n{'='*50}")
    print(f"Model Evaluation Results:")
    print(f"{'='*50}")
    print(f"log10(ID) MAE:  {mae:.4f}")
    print(f"log10(ID) RMSE: {rmse:.4f}")
    print(f"log10(ID) Max:  {max_error:.4f}")
    
    # 计算ID相对误差
    ids_pred = 10 ** all_preds
    ids_true = 10 ** all_targets
    
    # 只在ID > 1e-12的区域计算
    valid = ids_true > 1e-11
    if valid.sum() > 0:
        rel_error = np.abs(ids_pred[valid] - ids_true[valid]) / ids_true[valid]
        print(f"\nRelative Error (ID > 1e-11):")
        print(f"  Median: {np.median(rel_error)*100:.2f}%")
        print(f"  90th percentile: {np.percentile(rel_error, 90)*100:.2f}%")
        print(f"  95th percentile: {np.percentile(rel_error, 95)*100:.2f}%")
    
    return all_preds, all_targets, all_X


def plot_results(model, test_loader, train_losses, val_losses, device='cpu'):
    """绘制结果"""
    model.eval()
    
    all_X = []
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            pred = model(X).cpu()
            all_X.append(X.cpu().numpy())
            all_preds.append(pred.numpy())
            all_targets.append(y.numpy())
    
    all_X = np.concatenate(all_X)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 转移特性
    ax = axes[0, 0]
    for vds in [0.1, 0.3, 0.5, 0.8]:
        mask = np.abs(all_X[:, 1] - vds) < 0.02
        if mask.sum() > 10:
            ax.plot(all_X[mask, 0], all_targets[mask], 'o', markersize=2, 
                   label=f'True VDS={vds}V')
            ax.plot(all_X[mask, 0], all_preds[mask], '-', linewidth=1.5)
    ax.set_xlabel('VGS (V)')
    ax.set_ylabel('log10(ID)')
    ax.set_title('Transfer Characteristics')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 2. 输出特性
    ax = axes[0, 1]
    for vgs in [0.3, 0.5, 0.7, 0.9]:
        mask = np.abs(all_X[:, 0] - vgs) < 0.02
        if mask.sum() > 10:
            ids_true = 10 ** all_targets[mask]
            ids_pred = 10 ** all_preds[mask]
            ax.semilogy(all_X[mask, 1], ids_true, 'o', markersize=2, 
                       label=f'True VGS={vgs}V')
            ax.semilogy(all_X[mask, 1], ids_pred, '-', linewidth=1.5)
    ax.set_xlabel('VDS (V)')
    ax.set_ylabel('ID (A)')
    ax.set_title('Output Characteristics')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 3. 预测vs真实
    ax = axes[0, 2]
    ax.scatter(all_targets, all_preds, s=2, alpha=0.3)
    min_val = min(all_targets.min(), all_preds.min())
    max_val = max(all_targets.max(), all_preds.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
    ax.set_xlabel('True log10(ID)')
    ax.set_ylabel('Predicted log10(ID)')
    ax.set_title(f'Prediction vs Truth (MAE={np.mean(np.abs(all_preds-all_targets)):.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 误差分布
    ax = axes[1, 0]
    error = all_preds - all_targets
    ax.hist(error, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='r', linestyle='--')
    ax.axvline(np.mean(error), color='g', linestyle='--', label=f'Mean={np.mean(error):.4f}')
    ax.axvline(np.median(error), color='orange', linestyle='--', label=f'Median={np.median(error):.4f}')
    ax.set_xlabel('log10(ID) Error')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. 误差 vs VGS
    ax = axes[1, 1]
    for vds in [0.1, 0.3, 0.5, 0.8]:
        mask = np.abs(all_X[:, 1] - vds) < 0.02
        if mask.sum() > 10:
            ax.plot(all_X[mask, 0], error[mask], '-', linewidth=1, label=f'VDS={vds}V')
    ax.set_xlabel('VGS (V)')
    ax.set_ylabel('log10(ID) Error')
    ax.set_title('Error vs VGS')
    ax.axhline(0, color='r', linestyle='--')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 6. 训练损失
    ax = axes[1, 2]
    ax.semilogy(train_losses, label='Train')
    ax.semilogy(val_losses, label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fixed_nn_results.png', dpi=150)
    plt.show()
    print("\nResults saved to fixed_nn_results.png")


def main():
    print("="*60)
    print("Fixed BSIM-NN Model Training (32x32 network)")
    print("="*60)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 生成数据
    print("\nStep 1: Generating training data")
    df = generate_training_data(n_points=60)
    df.to_csv('fixed_training_data.csv', index=False)
    
    # 创建数据集
    dataset = MOSFETDataset(df)
    
    n = len(dataset)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    test_size = n - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    print(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # 创建模型
    print("\nStep 2: Creating model")
    model = FixedBSIM_NN(input_dim=3, hidden_size=32, num_layers=2)
    
    # 设置归一化统计量
    X_train = np.array([dataset[i][0].numpy() for i in range(len(dataset))])
    model.set_normalization_stats(X_train)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    
    # 训练
    print("\nStep 3: Training model (this will take ~5-10 minutes)")
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=1000, device=device)
    
    # 评估
    print("\nStep 4: Evaluating model")
    preds, targets, X_data = evaluate_model(model, test_loader, device)
    
    # 绘图
    print("\nStep 5: Plotting results")
    plot_results(model, test_loader, train_losses, val_losses, device)
    
    # 保存最终模型
    torch.save(model.state_dict(), 'fixed_nn_final.pth')
    print("\nModel saved to 'fixed_nn_final.pth'")
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()