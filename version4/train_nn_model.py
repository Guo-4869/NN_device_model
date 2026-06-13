# train_nn_model_fixed_v3.py
"""
最终修复版：使用对数变换和数据裁剪
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
    def __init__(self):
        super(ISRU, self).__init__()
    
    def forward(self, x):
        return x / torch.sqrt(1 + x * x + 1e-8)


class BSIM_NN_IV(nn.Module):
    """神经网络模型"""
    
    def __init__(self, 
                 input_dim: int = 3,
                 hidden_size: int = 32,
                 num_hidden_layers: int = 2,
                 activation: str = 'isru'):
        super(BSIM_NN_IV, self).__init__()
        
        self.input_dim = input_dim
        
        # 选择激活函数
        if activation == 'isru':
            self.activation = ISRU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        
        # 构建网络
        layers = []
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(self.activation)
        
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(self.activation)
            layers.append(nn.Dropout(0.1))  # 添加Dropout防止过拟合
        
        layers.append(nn.Linear(hidden_size, 1))
        
        self.network = nn.Sequential(*layers)
        
        # 归一化参数
        self.register_buffer('input_means', torch.zeros(input_dim))
        self.register_buffer('input_stds', torch.ones(input_dim))
        
    def forward(self, x):
        """前向传播 - 直接输出log10(ID)"""
        # 归一化输入
        x_norm = (x - self.input_means) / (self.input_stds + 1e-8)
        
        # 神经网络输出 log10(ID)
        log10_id = self.network(x_norm).squeeze()
        
        # 限制范围
        log10_id = torch.clamp(log10_id, -12, 0)
        
        # 计算ID
        ID = 10 ** log10_id
        
        return {'ID': ID, 'log10_ID': log10_id}
    
    def set_normalization(self, X_train):
        """设置归一化参数"""
        self.input_means = torch.tensor(X_train.mean(axis=0), dtype=torch.float32)
        self.input_stds = torch.tensor(X_train.std(axis=0) + 1e-6, dtype=torch.float32)


class ImprovedBSIM3Model:
    """改进的BSIM3模型 - 限制ID范围"""
    
    def __init__(self):
        self.tox = 2e-9
        self.cox = 3.9 * 8.854e-12 / self.tox
        self.vth0 = 0.45
        self.u0 = 0.04
        self.vsat = 8e4
        self.vt = 0.026
        
    def calc_ids(self, vgs, vds, vbs=0, l=1e-6, w=10e-6):
        """计算漏极电流 - 限制在合理范围"""
        vth = self.vth0 + 0.3 * (np.sqrt(0.8 - vbs) - np.sqrt(0.8))
        vth = max(vth, 0.1)
        
        vgst = vgs - vth
        if vgst <= 0:
            n = 1.2
            ids = 1e-6 * (w/l) * np.exp(vgst / (n * self.vt))  # 提高亚阈值电流
            ids *= (1 - np.exp(-vds / self.vt))
            return max(ids, 1e-12)
        
        ueff = self.u0 / (1 + 0.3 * vgst)
        esat = 2 * self.vsat / ueff
        vdsat = esat * l * vgst / (esat * l + vgst)
        beta = ueff * self.cox * (w / l)
        
        if vds < vdsat:
            ids = beta * (vgst - vds/2) * vds / (1 + vds/(esat*l))
        else:
            ids = beta * (vgst - vdsat/2) * vdsat / (1 + vdsat/(esat*l))
            ids *= (1 + 0.2 * (vds - vdsat))
        
        # 亚阈值贡献
        n = 1.2
        ids_sub = 1e-6 * (w/l) * np.exp((vgs - vth) / (n * self.vt))
        
        ids = ids + ids_sub
        
        # 限制ID范围在 1e-12 到 1e-3 之间
        return np.clip(ids, 1e-12, 1e-3)


def generate_training_data(l=1e-6, w=10e-6, n_points=60):
    """生成训练数据 - 使用对数均匀采样"""
    print("Generating training data...")
    
    model = ImprovedBSIM3Model()
    
    # 在线性空间采样VGS和VDS
    vgs = np.linspace(0, 1.0, n_points)
    vds = np.linspace(0, 1.0, n_points)
    
    data = []
    for vg in vgs:
        for vd in vds:
            id_val = model.calc_ids(vg, vd, 0, l, w)
            data.append({'VGS': vg, 'VDS': vd, 'VBS': 0, 'ID': id_val})
    
    df = pd.DataFrame(data)
    
    # 计算log10(ID)
    df['log10_ID'] = np.log10(df['ID'] + 1e-15)
    df['log10_ID'] = df['log10_ID'].clip(-12, -3)  # 限制范围
    
    # 计算导数（在对数域）
    df['gm_log'] = compute_gm_log(df)
    df['gds_log'] = compute_gds_log(df)
    
    print(f"Generated {len(df)} samples")
    print(f"ID range: {df['ID'].min():.2e} - {df['ID'].max():.2e} A")
    print(f"log10(ID) range: {df['log10_ID'].min():.2f} - {df['log10_ID'].max():.2f}")
    
    return df


def compute_gm_log(df):
    """计算跨导 (d(log10ID)/dVGS)"""
    gm_log = np.zeros(len(df))
    for vds in df['VDS'].unique():
        mask = df['VDS'] == vds
        idx = np.where(mask)[0]
        if len(idx) > 2:
            vgs_vals = df['VGS'].values[idx]
            logid_vals = df['log10_ID'].values[idx]
            sort_idx = np.argsort(vgs_vals)
            vgs_sorted = vgs_vals[sort_idx]
            logid_sorted = logid_vals[sort_idx]
            idx_sorted = idx[sort_idx]
            
            # 中心差分
            for i, pos in enumerate(idx_sorted):
                if i == 0:
                    gm_log[pos] = (logid_sorted[1] - logid_sorted[0]) / (vgs_sorted[1] - vgs_sorted[0] + 1e-8)
                elif i == len(idx_sorted) - 1:
                    gm_log[pos] = (logid_sorted[-1] - logid_sorted[-2]) / (vgs_sorted[-1] - vgs_sorted[-2] + 1e-8)
                else:
                    gm_log[pos] = (logid_sorted[i+1] - logid_sorted[i-1]) / (vgs_sorted[i+1] - vgs_sorted[i-1] + 1e-8)
    
    return np.clip(gm_log, -10, 10)


def compute_gds_log(df):
    """计算输出电导 (d(log10ID)/dVDS)"""
    gds_log = np.zeros(len(df))
    for vgs in df['VGS'].unique():
        mask = df['VGS'] == vgs
        idx = np.where(mask)[0]
        if len(idx) > 2:
            vds_vals = df['VDS'].values[idx]
            logid_vals = df['log10_ID'].values[idx]
            sort_idx = np.argsort(vds_vals)
            vds_sorted = vds_vals[sort_idx]
            logid_sorted = logid_vals[sort_idx]
            idx_sorted = idx[sort_idx]
            
            for i, pos in enumerate(idx_sorted):
                if i == 0:
                    gds_log[pos] = (logid_sorted[1] - logid_sorted[0]) / (vds_sorted[1] - vds_sorted[0] + 1e-8)
                elif i == len(idx_sorted) - 1:
                    gds_log[pos] = (logid_sorted[-1] - logid_sorted[-2]) / (vds_sorted[-1] - vds_sorted[-2] + 1e-8)
                else:
                    gds_log[pos] = (logid_sorted[i+1] - logid_sorted[i-1]) / (vds_sorted[i+1] - vds_sorted[i-1] + 1e-8)
    
    return np.clip(gds_log, -5, 5)


class MOSFETDataset(Dataset):
    """数据集"""
    def __init__(self, df):
        self.X = torch.tensor(df[['VGS', 'VDS', 'VBS']].values, dtype=torch.float32)
        self.log10_ID = torch.tensor(df['log10_ID'].values, dtype=torch.float32)
        self.ID = torch.tensor(df['ID'].values, dtype=torch.float32)
        self.gm_log = torch.tensor(df['gm_log'].values, dtype=torch.float32)
        self.gds_log = torch.tensor(df['gds_log'].values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {
            'X': self.X[idx],
            'log10_ID': self.log10_ID[idx],
            'ID': self.ID[idx],
            'gm_log': self.gm_log[idx],
            'gds_log': self.gds_log[idx]
        }


class CombinedLoss(nn.Module):
    """组合损失函数 - 重点在对数域"""
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        
    def forward(self, pred, target, compute_derivatives=False):
        """计算损失"""
        # 1. log10(ID) 损失 (主要)
        loss_log = self.mse(pred['log10_ID'], target['log10_ID'])
        
        # 2. ID相对误差损失
        rel_error = torch.abs((pred['ID'] - target['ID']) / (target['ID'] + 1e-12))
        loss_rel = torch.mean(torch.clamp(rel_error, max=0.5))
        
        # 3. 导数损失 (可选)
        loss_gm = 0
        loss_gds = 0
        if compute_derivatives:
            loss_gm = self.mse(pred['gm_log'], target['gm_log'])
            loss_gds = self.mse(pred['gds_log'], target['gds_log'])
        
        # 组合损失：主要关注对数域
        total_loss = loss_log + 0.1 * loss_rel + 0.05 * loss_gm + 0.05 * loss_gds
        
        return total_loss


def compute_derivatives_batch(model, X, eps=1e-4):
    """
    批量计算gm_log和gds_log
    """
    batch_size = X.shape[0]
    device = X.device
    
    # 复制输入用于扰动
    X_plus_vgs = X.clone()
    X_minus_vgs = X.clone()
    X_plus_vds = X.clone()
    X_minus_vds = X.clone()
    
    X_plus_vgs[:, 0] += eps
    X_minus_vgs[:, 0] -= eps
    X_plus_vds[:, 1] += eps
    X_minus_vds[:, 1] -= eps
    
    # 批量前向传播
    with torch.no_grad():
        log10_nom = model(X)['log10_ID']
        log10_plus_vgs = model(X_plus_vgs)['log10_ID']
        log10_minus_vgs = model(X_minus_vgs)['log10_ID']
        log10_plus_vds = model(X_plus_vds)['log10_ID']
        log10_minus_vds = model(X_minus_vds)['log10_ID']
    
    # 计算导数
    gm_log = (log10_plus_vgs - log10_minus_vgs) / (2 * eps)
    gds_log = (log10_plus_vds - log10_minus_vds) / (2 * eps)
    
    # 限制范围
    gm_log = torch.clamp(gm_log, -10, 10)
    gds_log = torch.clamp(gds_log, -5, 5)
    
    return gm_log, gds_log


def train_model(model, train_loader, val_loader, epochs=500, device='cuda'):
    """训练模型"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)
    criterion = CombinedLoss()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in tqdm(range(epochs), desc="Training"):
        # 训练阶段
        model.train()
        train_loss = 0
        train_batches = 0
        
        for batch in train_loader:
            X = batch['X'].to(device)
            targets = {
                'log10_ID': batch['log10_ID'].to(device),
                'ID': batch['ID'].to(device),
                'gm_log': batch['gm_log'].to(device),
                'gds_log': batch['gds_log'].to(device)
            }
            
            optimizer.zero_grad()
            
            # 前向传播
            predictions = model(X)
            
            # 计算导数
            gm_log, gds_log = compute_derivatives_batch(model, X)
            predictions['gm_log'] = gm_log
            predictions['gds_log'] = gds_log
            
            loss = criterion(predictions, targets, compute_derivatives=True)
            
            if torch.isnan(loss):
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        train_loss /= max(train_batches, 1)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                X = batch['X'].to(device)
                targets = {
                    'log10_ID': batch['log10_ID'].to(device),
                    'ID': batch['ID'].to(device)
                }
                
                predictions = model(X)
                loss = criterion(predictions, targets, compute_derivatives=False)
                
                if not torch.isnan(loss):
                    val_loss += loss.item()
                    val_batches += 1
        
        val_loss /= max(val_batches, 1)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        if (epoch + 1) % 100 == 0:
            print(f"\nEpoch {epoch+1}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
    
    return train_losses, val_losses


def evaluate_model(model, test_loader, device='cuda'):
    """评估模型"""
    model.eval()
    predictions = []
    targets = []
    log10_predictions = []
    log10_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            X = batch['X'].to(device)
            pred = model(X)
            predictions.extend(pred['ID'].cpu().numpy())
            targets.extend(batch['ID'].numpy())
            log10_predictions.extend(pred['log10_ID'].cpu().numpy())
            log10_targets.extend(batch['log10_ID'].numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    log10_predictions = np.array(log10_predictions)
    log10_targets = np.array(log10_targets)
    
    # 计算对数域误差
    log10_error = np.abs(log10_predictions - log10_targets)
    print(f"\nModel Evaluation:")
    print(f"  log10(ID) MAE: {np.mean(log10_error):.4f}")
    print(f"  log10(ID) Max Error: {np.max(log10_error):.4f}")
    
    # 计算相对误差
    valid_mask = targets > 1e-11
    if valid_mask.sum() > 0:
        rel_error = np.abs((predictions[valid_mask] - targets[valid_mask]) / targets[valid_mask])
        print(f"\n  Mean relative error: {np.mean(rel_error):.4f}")
        print(f"  Median relative error: {np.median(rel_error):.4f}")
        print(f"  90th percentile: {np.percentile(rel_error, 90):.4f}")
    
    return predictions, targets, log10_predictions, log10_targets


def plot_results(model, test_loader, train_losses, val_losses, device='cuda'):
    """绘制结果"""
    model.eval()
    
    # 收集数据
    all_X = []
    all_ID_true = []
    all_ID_pred = []
    all_log10_true = []
    all_log10_pred = []
    
    with torch.no_grad():
        for batch in test_loader:
            X = batch['X'].to(device)
            pred = model(X)
            all_X.append(batch['X'].numpy())
            all_ID_true.append(batch['ID'].numpy())
            all_ID_pred.append(pred['ID'].cpu().numpy())
            all_log10_true.append(batch['log10_ID'].numpy())
            all_log10_pred.append(pred['log10_ID'].cpu().numpy())
    
    all_X = np.concatenate(all_X)
    all_ID_true = np.concatenate(all_ID_true)
    all_ID_pred = np.concatenate(all_ID_pred)
    all_log10_true = np.concatenate(all_log10_true)
    all_log10_pred = np.concatenate(all_log10_pred)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. log10(ID)-VGS
    ax = axes[0, 0]
    for vds in [0.1, 0.3, 0.5, 0.8]:
        mask = np.abs(all_X[:, 1] - vds) < 0.02
        if mask.sum() > 10:
            ax.plot(all_X[mask, 0], all_log10_true[mask], 'o', markersize=2, label=f'True VDS={vds}V')
            ax.plot(all_X[mask, 0], all_log10_pred[mask], '-', linewidth=1, label=f'Pred VDS={vds}V')
    ax.set_xlabel('VGS (V)')
    ax.set_ylabel('log10(ID)')
    ax.set_title('Transfer Characteristics (log scale)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 2. ID-VDS
    ax = axes[0, 1]
    for vgs in [0.3, 0.5, 0.7, 0.9]:
        mask = np.abs(all_X[:, 0] - vgs) < 0.02
        if mask.sum() > 10:
            ax.semilogy(all_X[mask, 1], all_ID_true[mask], 'o', markersize=2, label=f'VGS={vgs}V')
            ax.semilogy(all_X[mask, 1], all_ID_pred[mask], '-', linewidth=1)
    ax.set_xlabel('VDS (V)')
    ax.set_ylabel('ID (A)')
    ax.set_title('Output Characteristics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 预测vs真实 (log10)
    ax = axes[0, 2]
    ax.scatter(all_log10_true, all_log10_pred, s=1, alpha=0.3)
    min_val = min(all_log10_true.min(), all_log10_pred.min())
    max_val = max(all_log10_true.max(), all_log10_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
    ax.set_xlabel('True log10(ID)')
    ax.set_ylabel('Predicted log10(ID)')
    ax.set_title('Prediction vs Truth')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 对数域误差分布
    ax = axes[1, 0]
    log10_error = np.abs(all_log10_pred - all_log10_true)
    ax.hist(log10_error, bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('log10(ID) Error')
    ax.set_ylabel('Frequency')
    ax.set_title('Log10 Error Distribution')
    ax.axvline(np.median(log10_error), color='r', linestyle='--', label=f'Median={np.median(log10_error):.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. 误差 vs VGS/VDS
    ax = axes[1, 1]
    error_pct = (all_log10_pred - all_log10_true)
    error_pct = np.clip(error_pct, -1, 1)
    scatter = ax.scatter(all_X[:, 0], all_X[:, 1], c=error_pct, cmap='RdBu', s=5, alpha=0.5, vmin=-0.5, vmax=0.5)
    ax.set_xlabel('VGS (V)')
    ax.set_ylabel('VDS (V)')
    ax.set_title('Log10 Error Map')
    plt.colorbar(scatter, ax=ax, label='log10 Error')
    
    # 6. 训练损失曲线
    ax = axes[1, 2]
    ax.semilogy(train_losses, label='Train')
    ax.semilogy(val_losses, label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nn_model_results.png', dpi=150)
    plt.show()


def main():
    print("="*60)
    print("Neural Network Compact Model Training (Final Version)")
    print("="*60)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # 生成数据
    print("\n" + "="*40)
    print("Step 1: Generating training data")
    print("="*40)
    
    df = generate_training_data(n_points=50)  # 2500个点
    
    # 保存数据
    df.to_csv('training_data.csv', index=False)
    
    # 创建数据集
    dataset = MOSFETDataset(df)
    
    # 划分数据集
    n = len(dataset)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    test_size = n - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # 创建模型
    print("\n" + "="*40)
    print("Step 2: Creating model")
    print("="*40)
    
    model = BSIM_NN_IV(input_dim=3, hidden_size=32, num_hidden_layers=2, activation='isru')
    
    # 设置归一化参数
    X_train = np.array([dataset[i]['X'].numpy() for i in range(len(dataset))])
    model.set_normalization(X_train)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    
    # 训练
    print("\n" + "="*40)
    print("Step 3: Training model")
    print("="*40)
    
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=500, device=device)
    
    # 评估
    print("\n" + "="*40)
    print("Step 4: Evaluating model")
    print("="*40)
    
    predictions, targets, log10_pred, log10_true = evaluate_model(model, test_loader, device)
    
    # 绘图
    print("\n" + "="*40)
    print("Step 5: Plotting results")
    print("="*40)
    
    plot_results(model, test_loader, train_losses, val_losses, device)
    
    # 保存模型
    torch.save(model.state_dict(), 'nn_model_final.pth')
    print("\nModel saved to 'nn_model_final.pth'")
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)


if __name__ == "__main__":
    main()