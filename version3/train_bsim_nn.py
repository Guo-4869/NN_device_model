# train_bsim_nn.py
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from bsim_nn_model import BSIMNNLoss, BSIM_NN_IV
class BSIMDataset(Dataset):
    """BSIM训练数据集"""
    
    def __init__(self, csv_file: str):
        data = pd.read_csv(csv_file)
        
        # 移除无效值
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        
        # 输入特征
        self.X = torch.tensor(data[['VGS', 'VDS', 'L', 'H_FIN', 'EOT']].values, 
                              dtype=torch.float32)
        
        # 目标值
        self.ID = torch.tensor(data['ID'].values, dtype=torch.float32)
        self.ID = torch.clamp(self.ID, min=1e-12)
        
        # 计算导数
        self.gm = self._compute_derivative_stable(data, 'VGS')
        self.gds = self._compute_derivative_stable(data, 'VDS')
        
    def _compute_derivative_stable(self, data, var_name):
        """稳定计算导数"""
        deriv = np.zeros(len(data))
        
        # 获取唯一的组合
        other_var = 'VDS' if var_name == 'VGS' else 'VGS'
        
        for l_val in data['L'].unique():
            for other_val in data[other_var].unique():
                mask = (data['L'] == l_val) & (data[other_var] == other_val)
                idx = np.where(mask)[0]
                
                if len(idx) > 1:
                    x_vals = data[var_name].values[idx]
                    y_vals = data['ID'].values[idx]
                    
                    # 排序
                    sort_idx = np.argsort(x_vals)
                    x_vals = x_vals[sort_idx]
                    y_vals = y_vals[sort_idx]
                    idx = idx[sort_idx]
                    
                    # 中心差分
                    for i, pos in enumerate(idx):
                        if i == 0:
                            dx = x_vals[1] - x_vals[0]
                            if dx > 1e-8:
                                deriv[pos] = (y_vals[1] - y_vals[0]) / dx
                        elif i == len(idx) - 1:
                            dx = x_vals[-1] - x_vals[-2]
                            if dx > 1e-8:
                                deriv[pos] = (y_vals[-1] - y_vals[-2]) / dx
                        else:
                            dx = x_vals[i+1] - x_vals[i-1]
                            if dx > 1e-8:
                                deriv[pos] = (y_vals[i+1] - y_vals[i-1]) / dx
        
        # 处理异常值
        deriv = np.nan_to_num(deriv, nan=0.0, posinf=1e-3, neginf=-1e-3)
        deriv = np.clip(deriv, -1e-2, 1e-2)
        
        return torch.tensor(deriv, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {
            'X': self.X[idx],
            'ID': self.ID[idx],
            'gm': self.gm[idx],
            'gds': self.gds[idx]
        }


def train_bsim_nn(model: torch.nn.Module, 
                  train_loader: DataLoader,
                  val_loader: DataLoader,
                  epochs: int = 200,
                  lr: float = 0.0005,
                  device: str = 'cuda'):
    """训练BSIM-NN模型"""
    
    model = model.to(device)
    
    # 使用 AdamW 优化器
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=20)
    
    criterion = BSIMNNLoss()
    
    train_losses = []
    val_losses = []
    
    # 梯度裁剪阈值
    max_grad_norm = 1.0
    
    for epoch in tqdm(range(epochs), desc="Training"):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch in train_loader:
            X = batch['X'].to(device)
            
            optimizer.zero_grad()
            
            try:
                # 前向传播
                predictions = model(X, compute_derivatives=True)
                targets = {
                    'ID': batch['ID'].to(device),
                    'gm': batch['gm'].to(device),
                    'gds': batch['gds'].to(device)
                }
                
                loss = criterion(predictions, targets)
                
                # 检查 loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN loss at epoch {epoch}, skipping batch")
                    continue
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
                
            except RuntimeError as e:
                print(f"RuntimeError at epoch {epoch}: {e}")
                continue
        
        if train_batches > 0:
            train_loss /= train_batches
        else:
            train_loss = float('nan')
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                X = batch['X'].to(device)
                
                try:
                    # 验证时不计算导数
                    predictions = model(X, compute_derivatives=False)
                    targets = {'ID': batch['ID'].to(device)}
                    
                    loss = criterion(predictions, targets)
                    
                    if not torch.isnan(loss) and not torch.isinf(loss):
                        val_loss += loss.item()
                        val_batches += 1
                except:
                    continue
        
        if val_batches > 0:
            val_loss /= val_batches
        else:
            val_loss = float('nan')
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # 更新学习率
        if not np.isnan(val_loss):
            scheduler.step(val_loss)
        
        # 打印进度
        if (epoch + 1) % 20 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, "
                  f"Val Loss = {val_loss:.6f}, LR = {current_lr:.6f}")
        
        # 早停检查
        if epoch > 50 and len(val_losses) > 20:
            recent_losses = val_losses[-20:]
            if np.min(recent_losses) > np.min(val_losses[:-20]) * 1.2:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return train_losses, val_losses


def generate_training_data():
    """生成平滑的训练数据"""
    print("Generating training data...")
    
    # 使用较粗的网格减少数据量
    VGS = np.linspace(0, 0.8, 30)
    VDS = np.linspace(0, 0.8, 30)
    L_vals = [18.0]
    H_FIN_vals = [46.0]
    EOT_vals = [0.78]
    
    data = []
    
    def smooth_mosfet_model(VG, VD, L=18, H_FIN=46, EOT=0.78):
        """简化的平滑MOSFET模型"""
        VTH = 0.3
        n = 1.2
        
        # 使用 softplus 实现平滑
        VGS_eff = np.log1p(np.exp(VG - VTH))
        
        # 饱和电压
        VDS_sat = VGS_eff / n
        
        # 增益因子
        beta = 1e-4 * (H_FIN / 46) * (18 / L)
        
        # 平滑过渡
        alpha_trans = 5.0
        ID_linear = beta * (VGS_eff * VD - n/2 * VD**2)
        ID_sat = beta * (VGS_eff**2) / (2*n)
        
        # 使用 sigmoid 作为平滑过渡
        transition = 1 / (1 + np.exp(-alpha_trans * (VD - VDS_sat)))
        ID = ID_linear * (1 - transition) + ID_sat * transition
        
        # 亚阈值电流
        ID_sub = 1e-7 * np.exp((VG - VTH) / (n * 0.026))
        
        return max(ID + ID_sub, 1e-14)
    
    total_samples = len(VGS) * len(VDS) * len(L_vals) * len(H_FIN_vals) * len(EOT_vals)
    print(f"Generating {total_samples} samples...")
    
    for L in L_vals:
        for H_FIN in H_FIN_vals:
            for EOT in EOT_vals:
                for VG in VGS:
                    for VD in VDS:
                        ID = smooth_mosfet_model(VG, VD, L, H_FIN, EOT)
                        
                        data.append({
                            'VGS': VG,
                            'VDS': VD,
                            'L': L,
                            'H_FIN': H_FIN,
                            'EOT': EOT,
                            'ID': ID
                        })
    
    df = pd.DataFrame(data)
    df.to_csv('training_data.csv', index=False)
    print(f"Generated {len(df)} training samples")
    print(f"ID range: {df['ID'].min():.2e} - {df['ID'].max():.2e}")
    
    return df


def visualize_results(model, test_loader, device='cuda'):
    """可视化训练结果"""
    model.eval()
    
    # 获取一批测试数据
    batch = next(iter(test_loader))
    X = batch['X'].to(device)
    
    with torch.no_grad():
        predictions = model(X, compute_derivatives=False)
    
    pred_id = predictions['ID'].cpu().numpy()
    target_id = batch['ID'].numpy()
    
    # 计算相对误差
    rel_error = np.abs((pred_id - target_id) / (target_id + 1e-12))
    
    print(f"Mean relative error: {np.mean(rel_error):.4f}")
    print(f"Max relative error: {np.max(rel_error):.4f}")
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 散点图
    axes[0].scatter(target_id, pred_id, alpha=0.5, s=1)
    axes[0].plot([1e-12, 1e-2], [1e-12, 1e-2], 'r--', label='Ideal')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Target ID (A)')
    axes[0].set_ylabel('Predicted ID (A)')
    axes[0].set_title('ID Prediction')
    axes[0].legend()
    
    # 误差分布
    axes[1].hist(rel_error[rel_error < 1], bins=50, alpha=0.7)
    axes[1].set_xlabel('Relative Error')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Error Distribution')
    
    plt.tight_layout()
    plt.savefig('model_validation.png')
    plt.show()


if __name__ == "__main__":
    print("=" * 50)
    print("BSIM-NN Training Script")
    print("=" * 50)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 生成或加载数据
    try:
        df = pd.read_csv('training_data.csv')
        print(f"Loaded existing data: {len(df)} samples")
    except FileNotFoundError:
        df = generate_training_data()
    
    # 创建数据集
    dataset = BSIMDataset('training_data.csv')
    
    # 划分数据集
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 创建模型 - 使用较小的网络
    model = BSIM_NN_IV(input_dim=5, hidden_dims=[16, 16])
    
    # 统计参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    
    # 训练
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    train_losses, val_losses = train_bsim_nn(
        model, train_loader, val_loader, 
        epochs=200, lr=0.0005, device=device
    )
    
    # 可视化结果
    visualize_results(model, test_loader, device)
    
    # 保存模型
    torch.save(model.state_dict(), 'bsim_nn_model.pth')
    print("Model saved to bsim_nn_model.pth")
    
    # 导出权重
    weights_dict = {}
    for name, param in model.named_parameters():
        weights_dict[name] = param.detach().cpu().numpy().tolist()
    
    import json
    with open('trained_weights.json', 'w') as f:
        json.dump(weights_dict, f, indent=2)
    print("Weights exported to trained_weights.json")
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('BSIM-NN Training Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.show()
    
    print("\nTraining completed!")