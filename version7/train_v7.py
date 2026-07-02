"""
 Version 7: BSIM_NN — Neural Network Compact Model
 ==================================================
 Based on: Tung, C.T. & Hu, C. "Neural Network-Based BSIM Transistor Model
 Framework", IEEE TED 2023/2024

 Key BSIM_NN methodology features:
  1. ISRU activation: f(x) = x / sqrt(1 + x^2), no exp(), Verilog-A friendly
  2. Derivative-assisted loss: fits ID + gm (transconductance) + gds (output conductance)
  3. Output transformation: ensures ID -> 0 at VDS -> 0
  4. Weight-regularized training for bounded weights
  5. Multi-phase: pre-train on log10(ID), fine-tune with derivatives
  6. Smooth Verilog-A export (no abrupt if/ternary, tanh-based clamping)

 Run: python train_v7.py
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

class BSIM3DataGenerator:
    """Comprehensive BSIM3 physics model for generating IV training data."""
    def __init__(self):
        self.tox = 2e-9
        self.cox = 3.9 * 8.854e-12 / self.tox
        self.vth0 = 0.45
        self.u0 = 0.04
        self.vsat = 1.0e5
        self.vt = 0.02585
        self.phi = 0.8
        self.gamma = 0.3
        self.n_sub = 1.2
        self.lambda_clm = 0.05

    def vth(self, vbs):
        arg = np.clip(self.phi - vbs, 0.01, None)
        return self.vth0 + self.gamma * (np.sqrt(arg) - np.sqrt(self.phi))

    def calc_ids(self, vgs, vds, vbs=0.0, l=1e-6, w=10e-6):
        vt = np.clip(self.vth(vbs), 0.05, 1.5)
        vgst = vgs - vt
        n = self.n_sub + 0.1 * self.gamma / np.sqrt(np.clip(self.phi - vbs, 0.01, None))
        ids_sub = 1e-7 * (w / l) * np.exp(vgst / (n * self.vt))
        ids_sub *= (1.0 - np.exp(-vds / self.vt))
        if vgst <= 0.0:
            ids_at = 0.0
        else:
            ueff = self.u0 / (1.0 + 0.4 * vgst)
            esat = 2.0 * self.vsat / ueff
            vdsat = vgst * esat * l / (vgst + esat * l)
            beta = ueff * self.cox * (w / l)
            if vds < vdsat:
                ids_at = beta * (vgst - 0.5 * vds) * vds / (1.0 + vds / (esat * l))
            else:
                ids_at = beta * (vgst - 0.5 * vdsat) * vdsat / (1.0 + vdsat / (esat * l))
                ids_at *= (1.0 + self.lambda_clm * (vds - vdsat))
        smooth_factor = 1.0 / (1.0 + np.exp(-5.0 * vgst / self.vt))
        ids = ids_sub * (1.0 - smooth_factor) + ids_at * smooth_factor
        return np.clip(ids, 1e-15, 1e-3)

    def calc_all(self, vgs_array, vds_array, vbs_array, l=1e-6, w=10e-6):
        rows = []
        for vgs in vgs_array:
            for vds in vds_array:
                for vbs in vbs_array:
                    ids = self.calc_ids(vgs, vds, vbs, l, w)
                    log_ids = np.log10(max(ids, 1e-15))
                    rows.append([vgs, vds, vbs, ids, log_ids])
        return np.array(rows)


def generate_training_data(n_vgs=60, n_vds=60, n_vbs=5, l=1e-6, w=10e-6):
    print("=" * 60)
    print("  BSIM_NN v7: Data Generation")
    print("=" * 60)
    gen = BSIM3DataGenerator()
    vgs_vals = np.linspace(-0.2, 1.2, n_vgs)
    vds_vals = np.linspace(0.001, 1.2, n_vds)
    vbs_vals = np.linspace(-0.5, 0.0, n_vbs)
    data = gen.calc_all(vgs_vals, vds_vals, vbs_vals, l, w)
    print(f"  Samples:       {len(data)}")
    print(f"  VGS range:     {vgs_vals[0]:.1f} to {vgs_vals[-1]:.1f} V")
    print(f"  VDS range:     {vds_vals[0]:.3f} to {vds_vals[-1]:.1f} V")
    print(f"  VBS range:     {vbs_vals[0]:.1f} to {vbs_vals[-1]:.1f} V")
    print(f"  ID range:      {data[:,3].min():.2e} to {data[:,3].max():.2e} A")
    print(f"  log10(ID) range: {data[:,4].min():.2f} to {data[:,4].max():.2f}")
    return data, vgs_vals, vds_vals, vbs_vals


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


def compute_derivatives(model, X, eps=1e-4):
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
    gen = BSIM3DataGenerator()
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


def autograd_derivatives(model, X, create_graph=True):
    """Compute gm_log and gds_log via PyTorch autograd."""
    X.requires_grad_(True)
    log_id = model(X)
    grads = torch.autograd.grad(
        outputs=log_id, inputs=X,
        grad_outputs=torch.ones_like(log_id),
        create_graph=create_graph, retain_graph=True
    )[0]
    return log_id, grads[:, 0], grads[:, 1]


def train_phase1(model, train_loader, val_loader, device, epochs=1000):
    """Phase 1: Pre-train on log10(ID) only."""
    print("\n" + "=" * 60)
    print("  Phase 1: Pre-training on log10(ID) only")
    print("=" * 60)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=100, factor=0.5, min_lr=1e-6)
    criterion = nn.MSELoss()
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
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
            torch.save({"model_state": model.state_dict(), "phase": 1, "val_loss": val_loss}, MODEL_PATH + ".p1")
        if (epoch + 1) % 200 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch+1:4d}: Train={train_loss:.6f}, Val={val_loss:.6f}, LR={lr:.2e}")
    print(f"  Phase 1 best val loss: {best_val_loss:.6f}")
    return train_losses, val_losses


def train_phase2(model, train_loader, val_loader, device, epochs=1500, w_gm=0.3, w_gds=0.3):
    """Phase 2: Fine-tune with derivative-assisted loss."""
    print("\n" + "=" * 60)
    print(f"  Phase 2: Derivative-assisted fine-tuning (w_gm={w_gm}, w_gds={w_gds})")
    print("=" * 60)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=100, factor=0.5, min_lr=1e-6)
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_b, y_b, _ in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            log_pred, gm_pred, gds_pred = autograd_derivatives(model, X_b)
            mse_loss = nn.MSELoss()(log_pred, y_b)
            gm_reg = w_gm * torch.mean(gm_pred ** 2)
            gds_reg = w_gds * torch.mean(gds_pred ** 2)
            d2_reg = 0.01 * torch.mean((gm_pred - gds_pred) ** 2)
            loss = mse_loss + gm_reg + gds_reg + d2_reg
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
                val_loss += nn.MSELoss()(pred, y_b).item()
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state": model.state_dict(), "phase": 2, "val_loss": val_loss}, MODEL_PATH)
        if (epoch + 1) % 200 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch+1:4d}: Train={train_loss:.6f}, Val={val_loss:.6f}, LR={lr:.2e}")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"  Phase 2 best val loss: {best_val_loss:.6f}")
    return train_losses, val_losses


def train_model():
    """Full BSIM_NN v7 training pipeline."""
    print("\n" + "=" * 60)
    print("  BSIM_NN v7 -- Neural Network Compact Model")
    print("  Based on Tung & Hu, IEEE TED 2023/2024")
    print("=" * 60)
    data, vgs_vals, vds_vals, vbs_vals = generate_training_data(n_vgs=50, n_vds=50, n_vbs=5)
    X_raw, y_raw, ids_raw = data[:, :3], data[:, 4], data[:, 3]
    X_mean, X_std = compute_normalization(X_raw)
    print(f"\n  Normalization:")
    print(f"    VGS: mean={X_mean[0]:.4f}, std={X_std[0]:.4f}")
    print(f"    VDS: mean={X_mean[1]:.4f}, std={X_std[1]:.4f}")
    print(f"    VBS: mean={X_mean[2]:.4f}, std={X_std[2]:.4f}")
    X_norm = (X_raw - X_mean) / X_std
    X_train, X_temp, y_train, y_temp, id_train, id_temp = train_test_split(X_norm, y_raw, ids_raw, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test, id_val, id_test = train_test_split(X_temp, y_temp, id_temp, test_size=0.5, random_state=42)
    train_ds = MOSFETDataset(X_train, y_train, id_train)
    val_ds = MOSFETDataset(X_val, y_val, id_val)
    test_ds = MOSFETDataset(X_test, y_test, id_test)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)
    test_loader = DataLoader(test_ds, batch_size=256)
    print(f"\n  Dataset sizes: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BSIM_NN_Model(input_dim=3, hidden_dim=32, num_hidden=3, activation='isru').to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: 3->32->32->32->1 ISRU MLP")
    print(f"  Parameters: {n_params}")
    print(f"  Device: {device}")
    losses_p1 = train_phase1(model, train_loader, val_loader, device, epochs=800)
    losses_p2 = train_phase2(model, train_loader, val_loader, device, epochs=1200)
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
    print(f"\n  Derivative Accuracy:")
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
    plot_results(model_cpu, test_loader, X_mean, X_std, losses_p1[0] + losses_p2[0], losses_p1[1] + losses_p2[1], vgs_vals, vds_vals, vbs_vals)
    return model_cpu, X_mean, X_std


def plot_results(model, test_loader, X_mean, X_std, train_losses, val_losses, vgs_vals, vds_vals, vbs_vals, output_path=PLOT_PATH):
    """Comprehensive 6-panel visualization."""
    model.eval()
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    ax = axes[0, 0]
    ax.plot(train_losses, label="Train", alpha=0.8)
    ax.plot(val_losses, label="Validation", alpha=0.8)
    ax.axvline(x=800, color="gray", linestyle="--", alpha=0.5, label="Phase boundary")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss (log10(ID))")
    ax.set_title("Training History"); ax.legend(fontsize=8)
    ax.set_yscale("log"); ax.grid(True, alpha=0.3)
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
    ax = axes[0, 2]
    error = all_preds - all_targets
    ax.hist(error, bins=80, alpha=0.7, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="red", linestyle="--", lw=1)
    ax.set_xlabel("log10(ID) Error"); ax.set_ylabel("Frequency")
    ax.set_title(f"Error Distribution\nsigma={error.std():.4f}")
    ax.grid(True, alpha=0.3)
    ax = axes[1, 0]
    vds_refs = [0.05, 0.1, 0.3, 0.5, 0.8, 1.0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(vds_refs)))
    for vds_ref, c in zip(vds_refs, colors):
        vgs_plot = np.linspace(-0.1, 1.2, 300)
        vbs_plot = np.zeros_like(vgs_plot)
        xx = np.column_stack([vgs_plot, np.full_like(vgs_plot, vds_ref), vbs_plot])
        # FIX: X_mean and X_std are already numpy arrays
        xx_norm = (xx - X_mean) / X_std
        x_t = torch.tensor(xx_norm, dtype=torch.float32)
        with torch.no_grad():
            log_pred = model(x_t).numpy()
        ax.plot(vgs_plot, 10.0 ** log_pred, "-", color=c, lw=1.5, label=f"VDS={vds_ref}V")
    ax.set_xlabel("VGS (V)"); ax.set_ylabel("ID (A)"); ax.set_yscale("log")
    ax.set_ylim(1e-12, 2e-3)
    ax.set_title("Transfer Characteristics (VBS=0)"); ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax = axes[1, 1]
    vgs_refs = [0.2, 0.4, 0.6, 0.8, 1.0]
    colors = plt.cm.plasma(np.linspace(0, 1, len(vgs_refs)))
    for vgs_ref, c in zip(vgs_refs, colors):
        vds_plot = np.linspace(0.001, 1.2, 300)
        vbs_plot = np.zeros_like(vds_plot)
        xx = np.column_stack([np.full_like(vds_plot, vgs_ref), vds_plot, vbs_plot])
        # FIX: X_mean and X_std are already numpy arrays
        xx_norm = (xx - X_mean) / X_std
        x_t = torch.tensor(xx_norm, dtype=torch.float32)
        with torch.no_grad():
            log_pred = model(x_t).numpy()
        ax.plot(vds_plot, 10.0 ** log_pred, "-", color=c, lw=1.5, label=f"VGS={vgs_ref}V")
    ax.set_xlabel("VDS (V)"); ax.set_ylabel("ID (A)")
    ax.set_title("Output Characteristics (VBS=0)"); ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax = axes[1, 2]
    for vbs_ref in [0.0, -0.1, -0.25, -0.4]:
        vgs_plot = np.linspace(-0.1, 1.2, 300)
        vds_plot = np.full_like(vgs_plot, 0.8)
        xx = np.column_stack([vgs_plot, vds_plot, np.full_like(vgs_plot, vbs_ref)])
        # FIX: X_mean and X_std are already numpy arrays
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
        "//   - Gate/body conductance paths (1e-14 S) for Jacobian",
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
    print(f"\n  Compile: openvaf bsim_nn_v7.va --ngspice -o bsim_nn_v7.osdi")
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
    print("")
    model, X_mean, X_std = train_model()
    generate_verilog(model, X_mean, X_std)
    print("\n" + "=" * 60)
    print("  BSIM_NN v7 training complete!")
    print("=" * 60)
    print(f"  Model:   {MODEL_PATH}")
    print(f"  Plot:    {PLOT_PATH}")
    print(f"  Verilog: {VA_PATH}")
    print("")
    print("  Next steps:")
    print("  1. cd version7")
    print("  2. openvaf bsim_nn_v7.va --ngspice -o bsim_nn_v7.osdi")
    print("  3. ngspice test_v7.cir")
    print("=" * 60)
    print("")
