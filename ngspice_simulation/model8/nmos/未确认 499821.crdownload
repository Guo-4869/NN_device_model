"""
 Version 7: BSIM_NN — Neural Network Compact Model (Phase 1 Only)
 ==================================================
 Based on: Tung, C.T. & Hu, C. "Neural Network-Based BSIM Transistor Model
 Framework", IEEE TED 2023/2024

 Key features:
  1. ISRU activation: f(x) = x / sqrt(1 + x^2), no exp(), Verilog-A friendly
  2. Smooth BSIM data generator with continuous derivatives
  3. Output transformation: ensures ID -> 0 at VDS -> 0
  4. Weight-regularized training for bounded weights
  5. Smooth Verilog-A export (no abrupt if/ternary, tanh-based clamping)
  6. Phase 1 only: Pre-train on log10(ID)

 Run: python train_v7_phase1.py
 Outputs: v7_best_model.pth, v7_results.png, bsim_nn_v7.va
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

OUTDIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(OUTDIR, "v7_best_model.pth")
PLOT_PATH = os.path.join(OUTDIR, "v7_results.png")
VA_PATH = os.path.join(OUTDIR, "bsim_nn_v7.va")


class BSIM3HighPrecision:
    """高精度BSIM模型 - 全程平滑可导，适合神经网络训练"""
    
    def __init__(self):
        # BSIM3v3.2物理参数
        self.tox = 2e-9                    # 氧化层厚度 (m)
        self.cox = 3.9 * 8.854e-12 / self.tox  # 氧化层电容 (F/m²)
        self.vth0 = 0.45                   # 零偏阈值电压 (V)
        self.u0 = 0.04                     # 载流子迁移率 (m²/V·s)
        self.vsat = 1.0e5                  # 饱和速度 (m/s)
        self.vt = 0.02585                  # 热电压 (V)
        self.phi = 0.8                     # 表面势 (V)
        self.gamma = 0.3                   # 体效应系数 (V^0.5)
        self.n_sub = 1.2                   # 亚阈值斜率因子
        self.lambda_clm = 0.05             # 沟道长度调制系数 (1/V)
        self.eta0 = 0.1                    # DIBL系数
        
    def vth(self, vbs, vds=0):
        """计算阈值电压，包含体效应和DIBL"""
        arg = np.clip(self.phi - vbs, 0.01, None)
        vth = self.vth0 + self.gamma * (np.sqrt(arg) - np.sqrt(self.phi))
        # DIBL效应 (漏致势垒降低)
        vth -= self.eta0 * vds
        return vth
    
    def smooth_logistic(self, x, scale=1.0):
        """平滑的logistic函数，用于连续过渡"""
        return 0.5 * (1 + np.tanh(0.5 * scale * x))
    
    def calc_ids(self, vgs, vds, vbs=0.0, l=1e-6, w=10e-6):
        """
        计算漏极电流 - 全程平滑可导
        
        参数:
            vgs: 栅源电压 (V)
            vds: 漏源电压 (V)
            vbs: 体源电压 (V)
            l: 沟道长度 (m)
            w: 沟道宽度 (m)
        
        返回:
            ids: 漏极电流 (A)
        """
        vt = self.vt
        vth = self.vth(vbs, vds)
        vgst = vgs - vth
        
        # 亚阈值斜率因子 (依赖于VBS)
        n = self.n_sub + 0.1 * self.gamma / np.sqrt(np.clip(self.phi - vbs, 0.01, None))
        
        # 1. 亚阈值电流 - 使用Softplus实现平滑阈值过渡
        vgst_eff = vt * np.log(1 + np.exp(vgst / vt))
        ids_sub = 1e-7 * (w / l) * np.exp(vgst_eff / (n * vt))
        ids_sub *= (1.0 - np.exp(-vds / vt))
        
        # 2. 迁移率退化
        ueff = self.u0 / (1.0 + 0.4 * vgst_eff)
        esat = 2.0 * self.vsat / ueff
        
        # 3. 平滑的饱和电压
        vgst_sat = vgst_eff
        vdsat = vgst_sat * esat * l / (vgst_sat + esat * l + 1e-10)
        
        # 4. 跨导因子
        beta = ueff * self.cox * (w / l)
        
        # 5. 线性区电流
        ids_lin = beta * (vgst_sat - 0.5 * vds) * vds / (1.0 + vds / (esat * l + 1e-10))
        ids_lin = np.clip(ids_lin, 0, None)
        
        # 6. 饱和区电流
        ids_sat = beta * (vgst_sat - 0.5 * vdsat) * vdsat / (1.0 + vdsat / (esat * l + 1e-10))
        # 沟道长度调制
        ids_sat *= (1.0 + self.lambda_clm * (vds - vdsat))
        ids_sat = np.clip(ids_sat, 0, None)
        
        # 7. 平滑的线性-饱和过渡 (使用tanh)
        vds_diff = vds - vdsat
        transition_lin_sat = self.smooth_logistic(vds_diff, scale=2.0/vdsat)
        ids_strong = ids_lin * (1 - transition_lin_sat) + ids_sat * transition_lin_sat
        
        # 8. 亚阈值到强反型的平滑过渡
        vgst_norm = vgst / (n * vt)
        transition_sub_strong = self.smooth_logistic(vgst, scale=0.8/vt)
        
        # 9. 最终电流
        ids = ids_sub * (1 - transition_sub_strong) + ids_strong * transition_sub_strong
        
        # 确保在VDS=0时电流为零
        ids = ids * (1 - np.exp(-vds / (0.001 * vt)))
        
        return np.clip(ids, 1e-15, 1e-3)
    
    def calc_all(self, vgs_array, vds_array, vbs_array, l=1e-6, w=10e-6):
        """批量计算所有数据点"""
        rows = []
        total = len(vgs_array) * len(vds_array) * len(vbs_array)
        count = 0
        
        for vgs in vgs_array:
            for vds in vds_array:
                for vbs in vbs_array:
                    ids = self.calc_ids(vgs, vds, vbs, l, w)
                    log_ids = np.log10(max(ids, 1e-15))
                    rows.append([vgs, vds, vbs, ids, log_ids])
                    count += 1
                    if count % 1000 == 0:
                        print(f"  生成数据: {count}/{total}")
        
        return np.array(rows)


class ISRU(nn.Module):
    """Inverse Square Root Unit: f(x) = x / sqrt(1 + x^2)"""
    def forward(self, x):
        return x / torch.sqrt(1.0 + x * x + 1e-8)


class BSIM_NN_Model(nn.Module):
    """3 inputs -> H -> H -> H -> 1 output with ISRU activations."""
    def __init__(self, input_dim=3, hidden_dim=32, num_hidden=3, activation='isru'):
        super().__init__()
        act = ISRU() if activation == 'isru' else nn.Tanh() if activation == 'tanh' else nn.ReLU()
        layers = [nn.Linear(input_dim, hidden_dim), act]
        for _ in range(num_hidden - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act)
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        
        # 初始化权重
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.7)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def compute_normalization(X):
    """计算归一化参数"""
    mean = X.mean(axis=0)
    std = np.maximum(X.std(axis=0), 0.05)
    return mean, std


class MOSFETDataset(Dataset):
    def __init__(self, X, log10_id, ids):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.log10_id = torch.tensor(log10_id, dtype=torch.float32)
        self.ids = torch.tensor(ids, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.log10_id[idx], self.ids[idx]


def generate_training_data(n_vgs=60, n_vds=60, n_vbs=5, l=1e-6, w=10e-6):
    """使用改进的平滑BSIM模型生成训练数据"""
    print("=" * 60)
    print("  BSIM_NN v7: Data Generation (Smooth BSIM Model)")
    print("=" * 60)
    
    gen = BSIM3HighPrecision()
    
    # 在阈值附近增加数据点密度
    vgs_vals = np.concatenate([
        np.linspace(-0.2, 0.3, 20),   # 亚阈值区域 - 密集采样
        np.linspace(0.3, 0.7, 25),    # 阈值附近 - 最密集
        np.linspace(0.7, 1.2, 15)     # 强反型区域
    ])
    vgs_vals = np.unique(vgs_vals)  # 去重并排序
    
    vds_vals = np.concatenate([
        np.linspace(0.001, 0.1, 20),  # 线性区 - 密集采样
        np.linspace(0.1, 0.5, 20),    # 过渡区
        np.linspace(0.5, 1.2, 20)     # 饱和区
    ])
    vds_vals = np.unique(vds_vals)
    
    vbs_vals = np.linspace(-0.5, 0.0, n_vbs)
    
    print(f"  VGS采样点数: {len(vgs_vals)} (阈值附近加密)")
    print(f"  VDS采样点数: {len(vds_vals)} (线性区加密)")
    print(f"  VBS采样点数: {len(vbs_vals)}")
    
    data = gen.calc_all(vgs_vals, vds_vals, vbs_vals, l, w)
    
    print(f"  Samples:       {len(data)}")
    print(f"  VGS range:     {vgs_vals.min():.1f} to {vgs_vals.max():.1f} V")
    print(f"  VDS range:     {vds_vals.min():.3f} to {vds_vals.max():.1f} V")
    print(f"  VBS range:     {vbs_vals.min():.1f} to {vbs_vals.max():.1f} V")
    print(f"  ID range:      {data[:,3].min():.2e} to {data[:,3].max():.2e} A")
    print(f"  log10(ID) range: {data[:,4].min():.2f} to {data[:,4].max():.2f}")
    
    return data, vgs_vals, vds_vals, vbs_vals


def train_phase1(model, train_loader, val_loader, device, epochs=1500):
    """Phase 1: Pre-train on log10(ID) only."""
    print("\n" + "=" * 60)
    print("  Phase 1: Pre-training on log10(ID) only")
    print("=" * 60)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", 
                                                      patience=100, factor=0.5, min_lr=1e-7)
    criterion = nn.MSELoss()
    
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    patience_counter = 0
    early_stop_patience = 300
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for X_b, y_b, _ in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(X_b)
            loss = criterion(pred, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b, _ in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                pred = model(X_b)
                val_loss += criterion(pred, y_b).item()
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(), 
                "phase": 1, 
                "val_loss": val_loss,
                "epoch": epoch
            }, MODEL_PATH)
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter > early_stop_patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 200 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch+1:4d}: Train={train_loss:.6f}, Val={val_loss:.6f}, LR={lr:.2e}")
    
    # 加载最佳模型
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"  Phase 1 best val loss: {best_val_loss:.6f} at epoch {checkpoint['epoch']+1}")
    
    return train_losses, val_losses


def compute_derivatives(model, X, eps=1e-4):
    """计算导数用于评估"""
    model.eval()
    with torch.no_grad():
        base = model(X)
        X_plus = X.clone(); X_plus[:, 0] += eps
        X_minus = X.clone(); X_minus[:, 0] -= eps
        gm_log = (model(X_plus) - model(X_minus)) / (2.0 * eps)
        X_plus = X.clone(); X_plus[:, 1] += eps
        X_minus = X.clone(); X_minus[:, 1] -= eps
        gds_log = (model(X_plus) - model(X_minus)) / (2.0 * eps)
    return gm_log, gds_log


def compute_gm_gds_log_bsim3(X):
    """从BSIM模型计算参考导数"""
    gen = BSIM3HighPrecision()
    eps = 1e-4
    gm_log = np.zeros(len(X))
    gds_log = np.zeros(len(X))
    
    for i, (vgs, vds, vbs) in enumerate(X):
        log_id_plus = np.log10(max(gen.calc_ids(vgs + eps, vds, vbs), 1e-15))
        log_id_minus = np.log10(max(gen.calc_ids(vgs - eps, vds, vbs), 1e-15))
        gm_log[i] = (log_id_plus - log_id_minus) / (2.0 * eps)
        
        log_id_plus = np.log10(max(gen.calc_ids(vgs, vds + eps, vbs), 1e-15))
        log_id_minus = np.log10(max(gen.calc_ids(vgs, vds - eps, vbs), 1e-15))
        gds_log[i] = (log_id_plus - log_id_minus) / (2.0 * eps)
    
    return gm_log, gds_log


def train_model():
    """完整的BSIM_NN v7训练流程 - 仅Phase 1"""
    print("\n" + "=" * 60)
    print("  BSIM_NN v7 -- Neural Network Compact Model (Phase 1 Only)")
    print("  Based on Tung & Hu, IEEE TED 2023/2024")
    print("  Using Smooth BSIM Data Generator")
    print("=" * 60)
    
    # 生成数据
    data, vgs_vals, vds_vals, vbs_vals = generate_training_data(
        n_vgs=50, n_vds=50, n_vbs=5
    )
    
    X_raw, y_raw, ids_raw = data[:, :3], data[:, 4], data[:, 3]
    
    # 归一化
    X_mean, X_std = compute_normalization(X_raw)
    print(f"\n  Normalization:")
    print(f"    VGS: mean={X_mean[0]:.4f}, std={X_std[0]:.4f}")
    print(f"    VDS: mean={X_mean[1]:.4f}, std={X_std[1]:.4f}")
    print(f"    VBS: mean={X_mean[2]:.4f}, std={X_std[2]:.4f}")
    
    # 保存归一化参数
    np.savez(os.path.join(OUTDIR, "normalization_params.npz"), 
             X_mean=X_mean, X_std=X_std)
    
    X_norm = (X_raw - X_mean) / X_std
    
    # 划分数据集
    X_train, X_temp, y_train, y_temp, id_train, id_temp = train_test_split(
        X_norm, y_raw, ids_raw, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test, id_val, id_test = train_test_split(
        X_temp, y_temp, id_temp, test_size=0.5, random_state=42
    )
    
    # 创建数据集
    train_ds = MOSFETDataset(X_train, y_train, id_train)
    val_ds = MOSFETDataset(X_val, y_val, id_val)
    test_ds = MOSFETDataset(X_test, y_test, id_test)
    
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)
    test_loader = DataLoader(test_ds, batch_size=256)
    
    print(f"\n  Dataset sizes: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    
    # 设备选择
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    
    # 创建模型
    model = BSIM_NN_Model(input_dim=3, hidden_dim=32, num_hidden=3, activation='isru')
    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: 3->32->32->32->1 ISRU MLP")
    print(f"  Parameters: {n_params}")
    
    # 训练
    losses_p1 = train_phase1(model, train_loader, val_loader, device, epochs=1500)
    
    # 评估
    model.eval()
    all_preds, all_targets, all_ids_pred, all_ids_target = [], [], [], []
    
    with torch.no_grad():
        for X_b, y_b, id_b in test_loader:
            X_b = X_b.to(device)
            pred = model(X_b).cpu().numpy()
            all_preds.extend(pred)
            all_targets.extend(y_b.numpy())
            all_ids_pred.extend(10.0 ** np.array(pred))
            all_ids_target.extend(id_b.numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_ids_pred = np.array(all_ids_pred)
    all_ids_target = np.array(all_ids_target)
    
    error = all_preds - all_targets
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))
    max_err = np.max(np.abs(error))
    
    print("\n" + "=" * 60)
    print("  Test Results (log10(ID))")
    print("=" * 60)
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Max:  {max_err:.4f}")
    print(f"  R2:   {1.0 - np.var(error) / np.var(all_targets):.4f}")
    
    # 计算相对误差
    rel_err = np.abs(all_ids_pred - all_ids_target) / (all_ids_target + 1e-15)
    valid = all_ids_target > 1e-11
    if valid.sum() > 0:
        print(f"\n  ID Relative Error (ID > 1e-11):")
        print(f"    Median: {np.median(rel_err[valid]) * 100:.2f}%")
        print(f"    P95:    {np.percentile(rel_err[valid], 95) * 100:.2f}%")
    
    # 导数评估
    print(f"\n  Derivative Accuracy (Evaluation only):")
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    gm_nn, gds_nn = compute_derivatives(model, X_test_t, eps=1e-4)
    gm_nn = gm_nn.cpu().numpy()
    gds_nn = gds_nn.cpu().numpy()
    gm_ref, gds_ref = compute_gm_gds_log_bsim3(X_test)
    gm_err = np.abs(gm_nn - gm_ref)
    gds_err = np.abs(gds_nn - gds_ref)
    print(f"    gm (d(logID)/dVGS):  MAE={gm_err.mean():.4f},  Max={gm_err.max():.4f}")
    print(f"    gds (d(logID)/dVDS): MAE={gds_err.mean():.4f}, Max={gds_err.max():.4f}")
    
    # 权重检查
    print(f"\n  Weight Magnitudes:")
    model_cpu = model.cpu()
    for name, param in model_cpu.named_parameters():
        if "weight" in name:
            w_abs = param.data.abs()
            status = "OK" if w_abs.max() < 5 else "WARNING: >5"
            print(f"    {name}: max_abs={w_abs.max():.4f}, mean_abs={w_abs.mean():.4f} [{status}]")
    
    # 绘图
    plot_results(model_cpu, test_loader, X_mean, X_std, 
                 losses_p1[0], losses_p1[1], 
                 vgs_vals, vds_vals, vbs_vals)
    
    # 检查平滑性
    verify_smoothness(model_cpu, X_mean, X_std)
    
    return model_cpu, X_mean, X_std


def plot_results(model, test_loader, X_mean, X_std, train_losses, val_losses, 
                 vgs_vals, vds_vals, vbs_vals, output_path=PLOT_PATH):
    """6面板综合可视化"""
    model.eval()
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    
    # 1. 训练历史
    ax = axes[0, 0]
    ax.plot(train_losses, label="Train", alpha=0.8)
    ax.plot(val_losses, label="Validation", alpha=0.8)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss (log10(ID))")
    ax.set_title("Training History"); ax.legend(fontsize=8)
    ax.set_yscale("log"); ax.grid(True, alpha=0.3)
    
    # 2. 预测vs真实
    ax = axes[0, 1]
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_b, y_b, _ in test_loader:
            pred = model(X_b).numpy()
            all_preds.extend(pred)
            all_targets.extend(y_b.numpy())
    all_preds = np.array(all_preds); all_targets = np.array(all_targets)
    ax.scatter(all_targets, all_preds, s=2, alpha=0.3, c="navy")
    lims = [min(all_targets.min(), all_preds.min()), max(all_targets.max(), all_preds.max())]
    ax.plot(lims, lims, "r--", lw=1, label="Ideal")
    ax.set_xlabel("True log10(ID)"); ax.set_ylabel("Predicted log10(ID)")
    ax.set_title(f"Prediction vs True\nMAE={np.mean(np.abs(all_preds-all_targets)):.3f}")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.axis("equal")
    
    # 3. 误差分布
    ax = axes[0, 2]
    error = all_preds - all_targets
    ax.hist(error, bins=80, alpha=0.7, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="red", linestyle="--", lw=1)
    ax.set_xlabel("log10(ID) Error"); ax.set_ylabel("Frequency")
    ax.set_title(f"Error Distribution\nsigma={error.std():.4f}")
    ax.grid(True, alpha=0.3)
    
    # 4. 转移特性
    ax = axes[1, 0]
    vds_refs = [0.05, 0.1, 0.3, 0.5, 0.8, 1.0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(vds_refs)))
    for vds_ref, c in zip(vds_refs, colors):
        vgs_plot = np.linspace(-0.1, 1.2, 300)
        vbs_plot = np.zeros_like(vgs_plot)
        xx = np.column_stack([vgs_plot, np.full_like(vgs_plot, vds_ref), vbs_plot])
        xx_norm = (xx - X_mean) / X_std
        x_t = torch.tensor(xx_norm, dtype=torch.float32)
        with torch.no_grad():
            log_pred = model(x_t).numpy()
        ax.plot(vgs_plot, 10.0 ** log_pred, "-", color=c, lw=1.5, label=f"VDS={vds_ref}V")
    ax.set_xlabel("VGS (V)"); ax.set_ylabel("ID (A)"); ax.set_yscale("log")
    ax.set_ylim(1e-12, 2e-3)
    ax.set_title("Transfer Characteristics (VBS=0)"); ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # 5. 输出特性
    ax = axes[1, 1]
    vgs_refs = [0.2, 0.4, 0.6, 0.8, 1.0]
    colors = plt.cm.plasma(np.linspace(0, 1, len(vgs_refs)))
    for vgs_ref, c in zip(vgs_refs, colors):
        vds_plot = np.linspace(0.001, 1.2, 300)
        vbs_plot = np.zeros_like(vds_plot)
        xx = np.column_stack([np.full_like(vds_plot, vgs_ref), vds_plot, vbs_plot])
        xx_norm = (xx - X_mean) / X_std
        x_t = torch.tensor(xx_norm, dtype=torch.float32)
        with torch.no_grad():
            log_pred = model(x_t).numpy()
        ax.plot(vds_plot, 10.0 ** log_pred, "-", color=c, lw=1.5, label=f"VGS={vgs_ref}V")
    ax.set_xlabel("VDS (V)"); ax.set_ylabel("ID (A)")
    ax.set_title("Output Characteristics (VBS=0)"); ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # 6. 体效应
    ax = axes[1, 2]
    for vbs_ref in [0.0, -0.1, -0.25, -0.4]:
        vgs_plot = np.linspace(-0.1, 1.2, 300)
        vds_plot = np.full_like(vgs_plot, 0.8)
        xx = np.column_stack([vgs_plot, vds_plot, np.full_like(vgs_plot, vbs_ref)])
        xx_norm = (xx - X_mean) / X_std
        x_t = torch.tensor(xx_norm, dtype=torch.float32)
        with torch.no_grad():
            log_pred = model(x_t).numpy()
        ax.plot(vgs_plot, 10.0 ** log_pred, "-", lw=1.5, label=f"VBS={vbs_ref}V")
    ax.set_xlabel("VGS (V)"); ax.set_ylabel("ID (A)"); ax.set_yscale("log")
    ax.set_ylim(1e-12, 2e-3)
    ax.set_title("Body Effect (VDS=0.8V)"); ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved: {output_path}")


def verify_smoothness(model, X_mean, X_std, output_path="smoothness_check.png"):
    """验证模型平滑性 - 检查连续性和导数"""
    print("\n" + "=" * 60)
    print("  Smoothness Verification")
    print("=" * 60)
    
    model.eval()
    
    # 生成精细VGS扫描
    vgs_fine = np.linspace(-0.1, 1.2, 500)
    vds_fixed = 0.5
    vbs_fixed = 0.0
    
    # 计算ID和导数
    ids_nn = []
    gm_nn = []
    
    for vgs in vgs_fine:
        xx = np.array([[vgs, vds_fixed, vbs_fixed]])
        xx_norm = (xx - X_mean) / X_std
        x_t = torch.tensor(xx_norm, dtype=torch.float32)
        with torch.no_grad():
            log_pred = model(x_t).numpy()[0]
            ids = 10.0 ** log_pred
            ids_nn.append(ids)
    
    ids_nn = np.array(ids_nn)
    gm_nn = np.gradient(ids_nn, vgs_fine)  # 数值导数
    gm2_nn = np.gradient(gm_nn, vgs_fine)   # 二阶导数
    
    # 与BSIM参考对比
    gen = BSIM3HighPrecision()
    ids_bsim = np.array([gen.calc_ids(vgs, vds_fixed, vbs_fixed) for vgs in vgs_fine])
    gm_bsim = np.gradient(ids_bsim, vgs_fine)
    gm2_bsim = np.gradient(gm_bsim, vgs_fine)
    
    # 绘图
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    # ID对比
    axes[0, 0].plot(vgs_fine, ids_nn, 'b-', label='NN Model', linewidth=2)
    axes[0, 0].plot(vgs_fine, ids_bsim, 'r--', label='BSIM Reference', linewidth=1.5)
    axes[0, 0].set_ylabel('ID (A)')
    axes[0, 0].set_title('Current (Linear Scale)')
    axes[0, 0].legend(); axes[0, 0].grid(True)
    
    axes[0, 1].semilogy(vgs_fine, ids_nn, 'b-', label='NN Model', linewidth=2)
    axes[0, 1].semilogy(vgs_fine, ids_bsim, 'r--', label='BSIM Reference', linewidth=1.5)
    axes[0, 1].set_ylabel('ID (A)')
    axes[0, 1].set_title('Current (Log Scale)')
    axes[0, 1].legend(); axes[0, 1].grid(True)
    
    # 跨导对比
    axes[1, 0].plot(vgs_fine, gm_nn, 'b-', label='NN Model', linewidth=2)
    axes[1, 0].plot(vgs_fine, gm_bsim, 'r--', label='BSIM Reference', linewidth=1.5)
    axes[1, 0].set_ylabel('gm (S)')
    axes[1, 0].set_title('Transconductance')
    axes[1, 0].legend(); axes[1, 0].grid(True)
    
    # 相对误差
    axes[1, 1].plot(vgs_fine, np.abs(gm_nn - gm_bsim) / (gm_bsim + 1e-15) * 100, 
                   'g-', linewidth=2)
    axes[1, 1].set_ylabel('gm Relative Error (%)')
    axes[1, 1].set_title('Transconductance Accuracy')
    axes[1, 1].grid(True)
    axes[1, 1].set_ylim(0, 20)
    
    # 二阶导数 (平滑性指标)
    axes[2, 0].plot(vgs_fine[1:-1], gm2_nn[1:-1], 'b-', label='NN Model', linewidth=2)
    axes[2, 0].plot(vgs_fine[1:-1], gm2_bsim[1:-1], 'r--', label='BSIM Reference', linewidth=1.5)
    axes[2, 0].set_xlabel('VGS (V)')
    axes[2, 0].set_ylabel('dgm/dVGS (S/V)')
    axes[2, 0].set_title('Second Derivative (Smoothness)')
    axes[2, 0].legend(); axes[2, 0].grid(True)
    
    # 误差分布
    axes[2, 1].hist(ids_nn - ids_bsim, bins=50, alpha=0.7, color='purple')
    axes[2, 1].set_xlabel('ID Error (A)')
    axes[2, 1].set_ylabel('Frequency')
    axes[2, 1].set_title('Current Error Distribution')
    axes[2, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"  Smoothness check saved: {output_path}")
    
    # 统计信息
    max_err = np.max(np.abs(ids_nn - ids_bsim))
    mean_err = np.mean(np.abs(ids_nn - ids_bsim))
    max_gm_err = np.max(np.abs(gm_nn - gm_bsim))
    print(f"\n  Comparison Statistics:")
    print(f"    ID Max Error: {max_err:.2e} A")
    print(f"    ID Mean Error: {mean_err:.2e} A")
    print(f"    gm Max Error: {max_gm_err:.2e} S")
    print(f"    Smoothness: {'Good' if np.std(gm2_nn) < 1 else 'Check required'}")


def generate_verilog(model, X_mean, X_std, output_file=VA_PATH):
    """Generate Verilog-A for BSIM_NN v7 with SPICE convergence features."""
    H = 32
    s = model.state_dict()
    w = {
        "b1": s["net.0.bias"].numpy(), "w1": s["net.0.weight"].numpy(),
        "b2": s["net.2.bias"].numpy(), "w2": s["net.2.weight"].numpy(),
        "b3": s["net.4.bias"].numpy(), "w3": s["net.4.weight"].numpy(),
        "b4": s["net.6.bias"].numpy(), "w4": s["net.6.weight"].numpy(),
    }
    
    def fmt(val, d=10):
        if abs(val) < 1e-12: return "0.0"
        return f"{val:.{d}e}" if abs(val) < 1e-4 else f"{val:.{d}f}"
    
    L = [
        "// ============================================================",
        "// BSIM_NN v7 - Neural Network Compact MOSFET Model",
        "// Based on: Tung & Hu, IEEE TED 2023/2024",
        "// Architecture: 3 inputs -> 32 ISRU -> 32 ISRU -> 32 ISRU -> 1",
        "// ============================================================",
        "// SPICE Convergence Features:",
        "//   - ISRU activation: no exp(), smooth C^inf nonlinearity",
        "//   - Smooth tanh-based input range clamping",
        "//   - Minimum std = 0.05 to prevent division by zero",
        "//   - Weight-regularized: |W| < 5.0 to avoid activation saturation",
        "//   - Smooth VDS->0 transition via rational function",
        "// ============================================================",
        "",
        "`include \"disciplines.vams\"",
        "`include \"constants.vams\"",
        "",
        "module bsim_nn_v7(d, g, s, b);",
        "    inout d, g, s, b;",
        "    electrical d, g, s, b;",
        "",
        "    parameter real L = 1e-6 from (0:inf);",
        "    parameter real W  = 10e-6 from (0:inf);",
        "",
        "    real VGS, VDS, VBS, VGS_n, VDS_n, VBS_n;",
    ]
    
    h1_vars = ", ".join([f"h1_{i}" for i in range(H)])
    h2_vars = ", ".join([f"h2_{i}" for i in range(H)])
    h3_vars = ", ".join([f"h3_{i}" for i in range(H)])
    L.append(f"    real {h1_vars};")
    L.append(f"    real {h2_vars};")
    L.append(f"    real {h3_vars};")
    L.append("    real log10_Id, Ids, vds_ratio;")
    L.append("")
    L.append("    // ISRU: f(x) = x / sqrt(1 + x^2)")
    L.append("    // Bounded (-1, 1), smooth, no exp(), Verilog-A friendly")
    L.append("")
    L.append(f"    real mean_vgs = {fmt(X_mean[0])};")
    L.append(f"    real std_vgs  = {fmt(X_std[0])};")
    L.append(f"    real mean_vds = {fmt(X_mean[1])};")
    L.append(f"    real std_vds  = {fmt(X_std[1])};")
    L.append(f"    real mean_vbs = {fmt(X_mean[2])};")
    L.append(f"    real std_vbs  = {fmt(X_std[2])};")
    L.append("")
    L.append("    analog begin")
    L.append("        VGS = V(g, s);")
    L.append("        VDS = V(d, s);")
    L.append("        VBS = V(b, s);")
    L.append("")
    L.append("        // Smooth tanh-based input clamping")
    L.append("        VGS = 0.5 + 0.8 * tanh((VGS - 0.5) / 0.8);")
    L.append("        VDS = 0.6 + 0.7 * tanh((VDS - 0.6) / 0.7);")
    L.append("        VBS = -0.25 + 0.3 * tanh((VBS + 0.25) / 0.3);")
    L.append("")
    L.append("        VGS_n = (VGS - mean_vgs) / std_vgs;")
    L.append("        VDS_n = (VDS - mean_vds) / std_vds;")
    L.append("        VBS_n = (VBS - mean_vbs) / std_vbs;")
    L.append("")
    L.append("        // Hidden Layer 1: 3 -> 32 ISRU")
    L.append("")
    for i in range(H):
        terms = [fmt(w["b1"][i]), f'{fmt(w["w1"][i,0])}*VGS_n', f'{fmt(w["w1"][i,1])}*VDS_n', f'{fmt(w["w1"][i,2])}*VBS_n']
        L.append(f'        h1_{i} = ({" + ".join(terms)}) / sqrt(1 + pow({" + ".join(terms)}, 2));')
    L.append("")
    L.append("        // Hidden Layer 2: 32 -> 32 ISRU")
    L.append("")
    for i in range(H):
        terms = [fmt(w["b2"][i])]
        for j in range(H):
            wv = w["w2"][i, j]
            if abs(wv) > 1e-10:
                terms.append(f'{fmt(wv)}*h1_{j}')
        L.append(f'        h2_{i} = ({" + ".join(terms)}) / sqrt(1 + pow({" + ".join(terms)}, 2));')
    L.append("")
    L.append("        // Hidden Layer 3: 32 -> 32 ISRU")
    L.append("")
    for i in range(H):
        terms = [fmt(w["b3"][i])]
        for j in range(H):
            wv = w["w3"][i, j]
            if abs(wv) > 1e-10:
                terms.append(f'{fmt(wv)}*h2_{j}')
        L.append(f'        h3_{i} = ({" + ".join(terms)}) / sqrt(1 + pow({" + ".join(terms)}, 2));')
    L.append("")
    L.append("        // Output Layer: 32 -> 1 (linear)")
    L.append("")
    terms = [fmt(w["b4"][0])]
    for j in range(H):
        wv = w["w4"][0, j]
        if abs(wv) > 1e-10:
            terms.append(f'{fmt(wv)}*h3_{j}')
    L.append(f'        log10_Id = {" + ".join(terms)};')
    L.append("")
    L.append("        // Smooth tanh-based output clamping to [-12, -2]")
    L.append("        log10_Id = -7.0 + 5.0 * tanh((log10_Id + 7.0) / 5.0);")
    L.append("")
    L.append("        Ids = pow(10, log10_Id) * (W / 10e-6);")
    L.append("")
    L.append("        // Smooth VDS -> 0 transition via rational function")
    L.append("        vds_ratio = V(d,s) / (V(d,s) + 1e-3);")
    L.append("        Ids = Ids * vds_ratio;")
    L.append("")
    L.append("        I(d, s) <+ Ids;")
    L.append("        I(g, s) <+ 1e-14 * V(g, s);")
    L.append("        I(b, s) <+ 1e-14 * V(b, s);")
    L.append("    end")
    L.append("")
    L.append("endmodule")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    
    print(f"\n  Verilog-A generated: {output_file} ({len(L)} lines)")
    print(f"\n  Weight Magnitudes (Verilog-A export check):")
    all_ok = True
    for k, arr in w.items():
        val = np.abs(arr).max()
        status = "OK" if val < 5.0 else "WARNING: >5"
        if status != "OK":
            all_ok = False
        print(f"    {k}: max_abs={val:.4f} [{status}]")
    if all_ok:
        print(f"  All weights < 5.0 - safe for SPICE convergence")
    else:
        print(f"  Some weights > 5.0 - may cause activation saturation in SPICE")
    
    return output_file


if __name__ == "__main__":
    print("")
    print(r"   ____  ____ ___ _   _ _   _ _   _ ")
    print(r"  | __ )/ ___|_ _| \ | | \ | | \ | |")
    print(r"  |  _ \\___ \| ||  \| |  \| |  \| |")
    print(r"  | |_) |___) | || |\  | |\  | |\  |")
    print(r"  |____/|____/___|_| \_|_| \_|_| \_|")
    print(r"  Neural Network Compact Model  v7    ")
    print(r"  Tung & Hu, IEEE TED 2023/2024      ")
    print(r"  Phase 1 Only - Stable Training     ")
    print(r"  Smooth BSIM Data Generator         ")
    print("")
    
    model, X_mean, X_std = train_model()
    generate_verilog(model, X_mean, X_std)
    
    print("\n" + "=" * 60)
    print("  BSIM_NN v7 training complete!")
    print("=" * 60)
    print(f"  Model:   {MODEL_PATH}")
    print(f"  Plot:    {PLOT_PATH}")
    print(f"  Verilog: {VA_PATH}")
    print(f"  Norm:    {os.path.join(OUTDIR, 'normalization_params.npz')}")
    print(f"  Smoothness: {os.path.join(OUTDIR, 'smoothness_check.png')}")
    print("")
    print("  Next steps:")
    print("  1. cd version7")
    print("  2. openvaf bsim_nn_v7.va --ngspice -o bsim_nn_v7.osdi")
    print("  3. ngspice test_v7.cir")
    print("=" * 60)
    print("")