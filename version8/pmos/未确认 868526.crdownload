"""
 Version 7: BSIM_NN — Neural Network Compact Model (PMOS Version)
 ==================================================
 Based on: Tung, C.T. & Hu, C. "Neural Network-Based BSIM Transistor Model
 Framework", IEEE TED 2023/2024

 PMOS Specific Features:
  1. All voltages negative (VGS, VDS, VBS < 0)
  2. Lower mobility (u0 = 0.02)
  3. Negative threshold voltage (vth0 = -0.45)
  4. Current flows from source to drain (positive ID for PMOS)
  5. ISRU activation: f(x) = x / sqrt(1 + x^2)

 Run: python train_pmos_v7.py
 Outputs: pmos_v7_best_model.pth, pmos_v7_results.png, bsim_pmos_v7.va
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
MODEL_PATH = os.path.join(OUTDIR, "pmos_v7_best_model.pth")
PLOT_PATH = os.path.join(OUTDIR, "pmos_v7_results.png")
VA_PATH = os.path.join(OUTDIR, "bsim_pmos_v7.va")


class BSIM3PMOSHighPrecision:
    """高精度PMOS模型 - 全程平滑可导，适合神经网络训练"""
    
    def __init__(self):
        # PMOS物理参数 (与NMOS对称，但极性相反)
        self.tox = 2e-9                    # 氧化层厚度 (m)
        self.cox = 3.9 * 8.854e-12 / self.tox  # 氧化层电容 (F/m²)
        self.vth0 = -0.45                  # 零偏阈值电压 (V) - PMOS为负
        self.u0 = 0.02                     # 空穴迁移率 (m²/V·s) - PMOS较低
        self.vsat = 0.8e5                  # 饱和速度 (m/s) - PMOS较低
        self.vt = 0.02585                  # 热电压 (V)
        self.phi = 0.8                     # 表面势 (V)
        self.gamma = 0.3                   # 体效应系数 (V^0.5)
        self.n_sub = 1.2                   # 亚阈值斜率因子
        self.lambda_clm = 0.05             # 沟道长度调制系数 (1/V)
        self.eta0 = 0.1                    # DIBL系数
        
    def vth(self, vbs, vds=0):
        """
        计算PMOS阈值电压
        注意：PMOS的VBS通常为正（相对于源极），VDS为负
        """
        # PMOS: vbs是正电压，体效应使阈值更负
        arg = np.clip(self.phi + vbs, 0.01, None)  # PMOS: phi + vbs
        vth = self.vth0 - self.gamma * (np.sqrt(arg) - np.sqrt(self.phi))
        # DIBL效应
        vth -= self.eta0 * vds  # vds为负，所以vth会变得更负
        return vth
    
    def smooth_logistic(self, x, scale=1.0):
        """平滑的logistic函数"""
        return 0.5 * (1 + np.tanh(0.5 * scale * x))
    
    def calc_ids(self, vgs, vds, vbs=0.0, l=1e-6, w=10e-6):
        """
        计算PMOS漏极电流（源极流出）
        
        参数:
            vgs: 栅源电压 (V) - 通常为负
            vds: 漏源电压 (V) - 通常为负
            vbs: 体源电压 (V) - 通常为正
            l: 沟道长度 (m)
            w: 沟道宽度 (m)
        
        返回:
            ids: 漏极电流 (A) - 正值（从源极流向漏极）
        """
        vt = self.vt
        
        # PMOS: 电压都是负的，取绝对值处理
        vgs_abs = -vgs
        vds_abs = -vds
        vbs_abs = vbs  # PMOS的VBS通常为正
        
        # 计算阈值电压 (PMOS)
        vth = self.vth(vbs, vds)
        vth_abs = -vth  # 取正用于计算
        
        # 过驱动电压 (PMOS)
        vgst = vgs_abs - vth_abs
        
        # 亚阈值斜率因子 (依赖于VBS)
        n = self.n_sub + 0.1 * self.gamma / np.sqrt(np.clip(self.phi + vbs, 0.01, None))
        
        # 1. 亚阈值电流 - 使用Softplus实现平滑阈值过渡
        vgst_eff = vt * np.log(1 + np.exp(vgst / vt))
        ids_sub = 1e-7 * (w / l) * np.exp(vgst_eff / (n * vt))
        ids_sub *= (1.0 - np.exp(-vds_abs / vt))
        
        # 2. 迁移率退化
        ueff = self.u0 / (1.0 + 0.4 * vgst_eff)
        esat = 2.0 * self.vsat / ueff
        
        # 3. 平滑的饱和电压
        vgst_sat = vgst_eff
        vdsat = vgst_sat * esat * l / (vgst_sat + esat * l + 1e-10)
        
        # 4. 跨导因子
        beta = ueff * self.cox * (w / l)
        
        # 5. 线性区电流
        ids_lin = beta * (vgst_sat - 0.5 * vds_abs) * vds_abs / (1.0 + vds_abs / (esat * l + 1e-10))
        ids_lin = np.clip(ids_lin, 0, None)
        
        # 6. 饱和区电流
        ids_sat = beta * (vgst_sat - 0.5 * vdsat) * vdsat / (1.0 + vdsat / (esat * l + 1e-10))
        # 沟道长度调制
        ids_sat *= (1.0 + self.lambda_clm * (vds_abs - vdsat))
        ids_sat = np.clip(ids_sat, 0, None)
        
        # 7. 平滑的线性-饱和过渡
        vds_diff = vds_abs - vdsat
        transition_lin_sat = self.smooth_logistic(vds_diff, scale=2.0/vdsat)
        ids_strong = ids_lin * (1 - transition_lin_sat) + ids_sat * transition_lin_sat
        
        # 8. 亚阈值到强反型的平滑过渡
        vgst_norm = vgst / (n * vt)
        transition_sub_strong = self.smooth_logistic(vgst, scale=0.8/vt)
        
        # 9. 最终电流 (PMOS电流为正)
        ids = ids_sub * (1 - transition_sub_strong) + ids_strong * transition_sub_strong
        
        # 确保在VDS=0时电流为零
        ids = ids * (1 - np.exp(-vds_abs / (0.001 * vt)))
        
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


class BSIM_PMOS_NN_Model(nn.Module):
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
        
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.7)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def compute_normalization(X):
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
    """使用平滑的PMOS模型生成训练数据"""
    print("=" * 60)
    print("  BSIM_PMOS v7: Data Generation (Smooth PMOS Model)")
    print("=" * 60)
    
    gen = BSIM3PMOSHighPrecision()
    
    # PMOS电压范围 (所有电压为负)
    vgs_vals = np.concatenate([
        np.linspace(0.2, -0.3, 20),    # 亚阈值区域 - 从正到负
        np.linspace(-0.3, -0.7, 25),   # 阈值附近 - 最密集
        np.linspace(-0.7, -1.2, 15)    # 强反型区域
    ])
    vgs_vals = np.sort(np.unique(vgs_vals))  # 排序（从正到负）
    
    vds_vals = np.concatenate([
        np.linspace(-0.001, -0.1, 20),  # 线性区 - 密集采样
        np.linspace(-0.1, -0.5, 20),    # 过渡区
        np.linspace(-0.5, -1.2, 20)     # 饱和区
    ])
    vds_vals = np.sort(np.unique(vds_vals))
    
    vbs_vals = np.linspace(0.0, 0.5, n_vbs)  # PMOS VBS为正
    
    print(f"  VGS采样点数: {len(vgs_vals)} (阈值附近加密)")
    print(f"  VDS采样点数: {len(vds_vals)} (线性区加密)")
    print(f"  VBS采样点数: {len(vbs_vals)}")
    print(f"  VGS范围: {vgs_vals[0]:.2f} to {vgs_vals[-1]:.2f} V")
    print(f"  VDS范围: {vds_vals[0]:.3f} to {vds_vals[-1]:.2f} V")
    print(f"  VBS范围: {vbs_vals[0]:.2f} to {vbs_vals[-1]:.2f} V")
    
    data = gen.calc_all(vgs_vals, vds_vals, vbs_vals, l, w)
    
    print(f"  Samples:       {len(data)}")
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
        
        if patience_counter > early_stop_patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 200 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch+1:4d}: Train={train_loss:.6f}, Val={val_loss:.6f}, LR={lr:.2e}")
    
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
    gen = BSIM3PMOSHighPrecision()
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
    """完整的BSIM_PMOS v7训练流程"""
    print("\n" + "=" * 60)
    print("  BSIM_PMOS v7 -- PMOS Neural Network Compact Model")
    print("  Based on Tung & Hu, IEEE TED 2023/2024")
    print("  Using Smooth PMOS Data Generator")
    print("=" * 60)
    
    data, vgs_vals, vds_vals, vbs_vals = generate_training_data(
        n_vgs=50, n_vds=50, n_vbs=5
    )
    
    X_raw, y_raw, ids_raw = data[:, :3], data[:, 4], data[:, 3]
    
    X_mean, X_std = compute_normalization(X_raw)
    print(f"\n  Normalization:")
    print(f"    VGS: mean={X_mean[0]:.4f}, std={X_std[0]:.4f}")
    print(f"    VDS: mean={X_mean[1]:.4f}, std={X_std[1]:.4f}")
    print(f"    VBS: mean={X_mean[2]:.4f}, std={X_std[2]:.4f}")
    
    np.savez(os.path.join(OUTDIR, "pmos_normalization_params.npz"), 
             X_mean=X_mean, X_std=X_std)
    
    X_norm = (X_raw - X_mean) / X_std
    
    X_train, X_temp, y_train, y_temp, id_train, id_temp = train_test_split(
        X_norm, y_raw, ids_raw, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test, id_val, id_test = train_test_split(
        X_temp, y_temp, id_temp, test_size=0.5, random_state=42
    )
    
    train_ds = MOSFETDataset(X_train, y_train, id_train)
    val_ds = MOSFETDataset(X_val, y_val, id_val)
    test_ds = MOSFETDataset(X_test, y_test, id_test)
    
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)
    test_loader = DataLoader(test_ds, batch_size=256)
    
    print(f"\n  Dataset sizes: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    
    model = BSIM_PMOS_NN_Model(input_dim=3, hidden_dim=32, num_hidden=3, activation='isru')
    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: 3->32->32->32->1 ISRU MLP")
    print(f"  Parameters: {n_params}")
    
    losses_p1 = train_phase1(model, train_loader, val_loader, device, epochs=1500)
    
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
    
    rel_err = np.abs(all_ids_pred - all_ids_target) / (all_ids_target + 1e-15)
    valid = all_ids_target > 1e-11
    if valid.sum() > 0:
        print(f"\n  ID Relative Error (ID > 1e-11):")
        print(f"    Median: {np.median(rel_err[valid]) * 100:.2f}%")
        print(f"    P95:    {np.percentile(rel_err[valid], 95) * 100:.2f}%")
    
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
    
    print(f"\n  Weight Magnitudes:")
    model_cpu = model.cpu()
    for name, param in model_cpu.named_parameters():
        if "weight" in name:
            w_abs = param.data.abs()
            status = "OK" if w_abs.max() < 5 else "WARNING: >5"
            print(f"    {name}: max_abs={w_abs.max():.4f}, mean_abs={w_abs.mean():.4f} [{status}]")
    
    plot_results_pmos(model_cpu, test_loader, X_mean, X_std, 
                      losses_p1[0], losses_p1[1], 
                      vgs_vals, vds_vals, vbs_vals)
    
    return model_cpu, X_mean, X_std


def plot_results_pmos(model, test_loader, X_mean, X_std, train_losses, val_losses, 
                      vgs_vals, vds_vals, vbs_vals, output_path=PLOT_PATH):
    """PMOS结果可视化"""
    model.eval()
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    
    # 1. Training History
    ax = axes[0, 0]
    ax.plot(train_losses, label="Train", alpha=0.8)
    ax.plot(val_losses, label="Validation", alpha=0.8)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss (log10(ID))")
    ax.set_title("PMOS Training History"); ax.legend(fontsize=8)
    ax.set_yscale("log"); ax.grid(True, alpha=0.3)
    
    # 2. Prediction vs True
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
    ax.set_title(f"PMOS Prediction vs True\nMAE={np.mean(np.abs(all_preds-all_targets)):.3f}")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.axis("equal")
    
    # 3. Error Distribution
    ax = axes[0, 2]
    error = all_preds - all_targets
    ax.hist(error, bins=80, alpha=0.7, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="red", linestyle="--", lw=1)
    ax.set_xlabel("log10(ID) Error"); ax.set_ylabel("Frequency")
    ax.set_title(f"PMOS Error Distribution\nsigma={error.std():.4f}")
    ax.grid(True, alpha=0.3)
    
    # 4. Transfer Characteristics (PMOS: VGS negative)
    ax = axes[1, 0]
    vds_refs = [-0.05, -0.1, -0.3, -0.5, -0.8, -1.0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(vds_refs)))
    for vds_ref, c in zip(vds_refs, colors):
        vgs_plot = np.linspace(0.2, -1.2, 300)
        vbs_plot = np.zeros_like(vgs_plot)
        xx = np.column_stack([vgs_plot, np.full_like(vgs_plot, vds_ref), vbs_plot])
        xx_norm = (xx - X_mean) / X_std
        x_t = torch.tensor(xx_norm, dtype=torch.float32)
        with torch.no_grad():
            log_pred = model(x_t).numpy()
        ax.plot(vgs_plot, 10.0 ** log_pred, "-", color=c, lw=1.5, label=f"VDS={vds_ref}V")
    ax.set_xlabel("VGS (V)"); ax.set_ylabel("ID (A)"); ax.set_yscale("log")
    ax.set_ylim(1e-12, 2e-3)
    ax.set_title("PMOS Transfer Characteristics (VBS=0)"); ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.2, -1.2)  # PMOS VGS从正到负
    
    # 5. Output Characteristics (PMOS: VDS negative)
    ax = axes[1, 1]
    vgs_refs = [-0.2, -0.4, -0.6, -0.8, -1.0]
    colors = plt.cm.plasma(np.linspace(0, 1, len(vgs_refs)))
    for vgs_ref, c in zip(vgs_refs, colors):
        vds_plot = np.linspace(-0.001, -1.2, 300)
        vbs_plot = np.zeros_like(vds_plot)
        xx = np.column_stack([np.full_like(vds_plot, vgs_ref), vds_plot, vbs_plot])
        xx_norm = (xx - X_mean) / X_std
        x_t = torch.tensor(xx_norm, dtype=torch.float32)
        with torch.no_grad():
            log_pred = model(x_t).numpy()
        ax.plot(vds_plot, 10.0 ** log_pred, "-", color=c, lw=1.5, label=f"VGS={vgs_ref}V")
    ax.set_xlabel("VDS (V)"); ax.set_ylabel("ID (A)")
    ax.set_title("PMOS Output Characteristics (VBS=0)"); ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.001, -1.2)
    
    # 6. Body Effect (PMOS: VBS positive)
    ax = axes[1, 2]
    for vbs_ref in [0.0, 0.1, 0.25, 0.4]:
        vgs_plot = np.linspace(0.2, -1.2, 300)
        vds_plot = np.full_like(vgs_plot, -0.8)
        xx = np.column_stack([vgs_plot, vds_plot, np.full_like(vgs_plot, vbs_ref)])
        xx_norm = (xx - X_mean) / X_std
        x_t = torch.tensor(xx_norm, dtype=torch.float32)
        with torch.no_grad():
            log_pred = model(x_t).numpy()
        ax.plot(vgs_plot, 10.0 ** log_pred, "-", lw=1.5, label=f"VBS={vbs_ref}V")
    ax.set_xlabel("VGS (V)"); ax.set_ylabel("ID (A)"); ax.set_yscale("log")
    ax.set_ylim(1e-12, 2e-3)
    ax.set_title("PMOS Body Effect (VDS=-0.8V)"); ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.2, -1.2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  PMOS Plot saved: {output_path}")


def generate_pmos_verilog(model, X_mean, X_std, output_file=VA_PATH):
    """生成PMOS Verilog-A代码"""
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
        "// BSIM_PMOS v7 - PMOS Neural Network Compact Model",
        "// Based on: Tung & Hu, IEEE TED 2023/2024",
        "// Architecture: 3 inputs -> 32 ISRU -> 32 ISRU -> 32 ISRU -> 1",
        "// ============================================================",
        "// PMOS Specific: All voltages negative",
        "// ============================================================",
        "",
        "`include \"disciplines.vams\"",
        "`include \"constants.vams\"",
        "",
        "module bsim_pmos_v7(d, g, s, b);",
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
    L.append("        // PMOS: voltage clamping for negative voltages")
    L.append("        VGS = -0.5 + 0.8 * tanh((VGS + 0.5) / 0.8);")
    L.append("        VDS = -0.6 + 0.7 * tanh((VDS + 0.6) / 0.7);")
    L.append("        VBS = 0.25 + 0.3 * tanh((VBS - 0.25) / 0.3);")
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
    L.append("        // Smooth tanh-based output clamping")
    L.append("        log10_Id = -7.0 + 5.0 * tanh((log10_Id + 7.0) / 5.0);")
    L.append("")
    L.append("        Ids = pow(10, log10_Id) * (W / 10e-6);")
    L.append("")
    L.append("        // Smooth VDS -> 0 transition")
    L.append("        vds_ratio = -V(d,s) / (-V(d,s) + 1e-3);")
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
    
    print(f"\n  PMOS Verilog-A generated: {output_file} ({len(L)} lines)")
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
    print(r"  PMOS Neural Network Model  v7      ")
    print(r"  Phase 1 Only - Stable Training     ")
    print(r"  Smooth PMOS Data Generator         ")
    print("")
    
    model, X_mean, X_std = train_model()
    generate_pmos_verilog(model, X_mean, X_std, VA_PATH)
    
    print("\n" + "=" * 60)
    print("  BSIM_PMOS v7 training complete!")
    print("=" * 60)
    print(f"  Model:   {MODEL_PATH}")
    print(f"  Plot:    {PLOT_PATH}")
    print(f"  Verilog: {VA_PATH}")
    print(f"  Norm:    {os.path.join(OUTDIR, 'pmos_normalization_params.npz')}")
    print("")
    print("  Next steps:")
    print("  1. cd version7")
    print("  2. openvaf bsim_pmos_v7.va --ngspice -o bsim_pmos_v7.osdi")
    print("  3. ngspice test_pmos_v7.cir")
    print("=" * 60)
    print("")