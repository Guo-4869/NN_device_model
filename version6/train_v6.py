"""
Version 6: SPICE-Compatible NN Compact Model Training
Run: python train_v6.py
Output: v6_best_model.pth, v6_results.png, bsim_nn_v6.va
"""
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, warnings, os
warnings.filterwarnings("ignore")
torch.manual_seed(42); np.random.seed(42)
OUTDIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(OUTDIR, "v6_best_model.pth")
PLOT_PATH = os.path.join(OUTDIR, "v6_results.png")
VA_PATH = os.path.join(OUTDIR, "bsim_nn_v6.va")

class BSIM3DataGenerator:
    def __init__(self):
        self.tox = 2e-9; self.cox = 3.9 * 8.854e-12 / self.tox
        self.vth0 = 0.45; self.u0 = 0.04; self.vsat = 8e4
        self.vt = 0.026; self.phi = 0.8; self.gamma = 0.3
    def vth(self, vbs):
        arg = np.clip(self.phi - vbs, 0.01, None)
        return self.vth0 + self.gamma * (np.sqrt(arg) - np.sqrt(self.phi))
    def calc_ids(self, vgs, vds, vbs=0, l=1e-6, w=10e-6):
        vt = np.clip(self.vth(vbs), 0.05, 1.5); vgst = vgs - vt
        if vgst <= 0:
            n = 1.3 + 0.1 * self.gamma / np.sqrt(np.clip(self.phi - vbs, 0.01, None))
            ids = 1e-7 * (w/l) * np.exp(vgst / (n * self.vt))
            ids *= (1 - np.exp(-vds / self.vt))
            return np.clip(ids, 1e-15, 1e-3)
        ueff = self.u0 / (1 + 0.3 * vgst); esat = 2 * self.vsat / ueff
        vdsat = vgst * esat * l / (vgst + esat * l); beta = ueff * self.cox * (w / l)
        if vds < vdsat:
            ids = beta * (vgst - vds/2) * vds / (1 + vds / (esat * l))
        else:
            ids = beta * (vgst - vdsat/2) * vdsat / (1 + vdsat / (esat * l))
            ids *= (1 + 0.1 * (vds - vdsat))
        n = 1.3
        ids_sub = 1e-7 * (w/l) * np.exp(vgst / (n * self.vt))
        ids_sub *= (1 - np.exp(-vds / self.vt)); ids = ids + ids_sub
        return np.clip(ids, 1e-15, 1e-3)

def generate_training_data(n_vgs=50, n_vds=50, n_vbs=5, l=1e-6, w=10e-6):
    print("Generating training data with body effect..."); gen = BSIM3DataGenerator()
    vgs_vals = np.linspace(-0.2, 1.2, n_vgs)
    vds_vals = np.linspace(0.001, 1.2, n_vds)
    vbs_vals = np.linspace(-0.5, 0.0, n_vbs)
    rows = []
    for vgs in vgs_vals:
        for vds in vds_vals:
            for vbs in vbs_vals:
                ids = gen.calc_ids(vgs, vds, vbs, l, w)
                log_ids = np.log10(max(ids, 1e-15))
                rows.append([vgs, vds, vbs, ids, log_ids])
    data = np.array(rows)
    print(f"Generated {len(data)} samples")
    print(f"  ID range: {data[:,3].min():.2e} - {data[:,3].max():.2e} A")
    print(f"  log10(ID) range: {data[:,4].min():.2f} - {data[:,4].max():.2f}")
    print(f"  VBS range: {data[:,2].min():.2f} - {data[:,2].max():.2f}")
    return data

class RegularizedNN(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim); self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        for m in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(m.weight, gain=0.7); nn.init.zeros_(m.bias)
    def forward(self, x):
        h = torch.tanh(self.fc1(x)); h = torch.tanh(self.fc2(h))
        return self.fc3(h).squeeze(-1)

class MOSFETDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32); self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def compute_normalization(X):
    mean = X.mean(axis=0); std = X.std(axis=0)
    return mean, np.maximum(std, 0.05)

def train_model():
    print("="*60)
    print("Version 6: SPICE-Compatible NN Compact Model Training")
    print("="*60)
    data = generate_training_data(n_vgs=50, n_vds=50, n_vbs=5)
    X_raw, y_raw = data[:, :3], data[:, 4]
    X_mean, X_std = compute_normalization(X_raw)
    print(f"\nNormalization stats:")
    print(f"  VGS: mean={X_mean[0]:.4f}, std={X_std[0]:.4f}")
    print(f"  VDS: mean={X_mean[1]:.4f}, std={X_std[1]:.4f}")
    print(f"  VBS: mean={X_mean[2]:.4f}, std={X_std[2]:.4f}")
    X_norm = (X_raw - X_mean) / X_std
    X_train, X_temp, y_train, y_temp = train_test_split(X_norm, y_raw, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    train_ds = MOSFETDataset(X_train, y_train)
    val_ds = MOSFETDataset(X_val, y_val); test_ds = MOSFETDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)
    test_loader = DataLoader(test_ds, batch_size=256)
    print(f"\nTrain: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RegularizedNN(input_dim=3, hidden_dim=32).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters())} parameters, device={device}")
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=200, factor=0.5, min_lr=1e-6)
    criterion = nn.MSELoss()
    epochs = 1500; train_losses, val_losses = [], []; best_val_loss = float("inf")
    print("\nTraining...")
    for epoch in range(epochs):
        model.train(); train_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device); optimizer.zero_grad()
            pred = model(X_b); loss = criterion(pred, y_b)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); train_loss += loss.item()
        train_loss /= len(train_loader)
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                pred = model(X_b); val_loss += criterion(pred, y_b).item()
        val_loss /= len(val_loader)
        train_losses.append(train_loss); val_losses.append(val_loss)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state": model.state_dict(), "X_mean": X_mean, "X_std": X_std}, MODEL_PATH)
        if (epoch + 1) % 300 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch+1:4d}: Train={train_loss:.6f}, Val={val_loss:.6f}, LR={lr:.2e}")
    print(f"\nBest validation loss: {best_val_loss:.6f}")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"]); model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b = X_b.to(device); pred = model(X_b).cpu().numpy()
            all_preds.extend(pred); all_targets.extend(y_b.numpy())
    all_preds = np.array(all_preds); all_targets = np.array(all_targets)
    error = all_preds - all_targets
    mae = np.mean(np.abs(error)); rmse = np.sqrt(np.mean(error**2)); max_err = np.max(np.abs(error))
    print(f"\nTest Results (log10(ID)):")
    print(f"  MAE:  {mae:.4f}"); print(f"  RMSE: {rmse:.4f}"); print(f"  Max:  {max_err:.4f}")
    ids_pred = 10**all_preds; ids_true = 10**all_targets
    rel_err = np.abs(ids_pred - ids_true) / (ids_true + 1e-15); valid = ids_true > 1e-11
    if valid.sum() > 0:
        print(f"\nID Relative Error (ID > 1e-11):")
        print(f"  Median: {np.median(rel_err[valid])*100:.2f}%")
        print(f"  P95:    {np.percentile(rel_err[valid],95)*100:.2f}%")
        print(f"  Max:    {np.max(rel_err[valid])*100:.2f}%")
    print("\nWeight magnitudes:")
    model_cpu = model.cpu()
    for name, param in model_cpu.named_parameters():
        if "weight" in name:
            w_abs = param.data.abs(); status = "OK" if w_abs.max() < 5 else "WARNING: >5"
            print(f"  {name}: max_abs={w_abs.max():.4f}, mean_abs={w_abs.mean():.4f} [{status}]")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes[0,0].semilogy(train_losses, label="Train"); axes[0,0].semilogy(val_losses, label="Validation")
    axes[0,0].set_xlabel("Epoch"); axes[0,0].set_ylabel("Loss (MSE)"); axes[0,0].set_title("Training History")
    axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)
    axes[0,1].scatter(all_targets, all_preds, s=1, alpha=0.3)
    mn, mx = all_targets.min(), all_targets.max()
    axes[0,1].plot([mn,mx],[mn,mx],"r--",lw=1)
    axes[0,1].set_xlabel("True log10(ID)"); axes[0,1].set_ylabel("Predicted log10(ID)")
    axes[0,1].set_title(f"MAE={mae:.3f}"); axes[0,1].grid(True, alpha=0.3)
    axes[0,2].hist(error, bins=60, alpha=0.7, edgecolor="black")
    axes[0,2].axvline(0, color="r", linestyle="--")
    axes[0,2].set_xlabel("log10(ID) Error"); axes[0,2].set_title("Error Distribution"); axes[0,2].grid(True, alpha=0.3)
    ax = axes[1,0]
    for vds_ref, c in zip([0.1,0.3,0.5,0.8,1.0], plt.cm.viridis(np.linspace(0,1,5))):
        vgs_plot = np.linspace(-0.2, 1.2, 200)
        xx = np.column_stack([vgs_plot, np.full_like(vgs_plot, vds_ref), np.zeros_like(vgs_plot)])
        x_t = torch.tensor((xx - X_mean) / X_std, dtype=torch.float32)
        with torch.no_grad(): y_pred = model_cpu(x_t).numpy()
        ax.plot(vgs_plot, 10**y_pred, "-", color=c, lw=1.5, label=f"VDS={vds_ref}V")
    ax.set_xlabel("VGS (V)"); ax.set_ylabel("ID (A)"); ax.set_yscale("log")
    ax.set_title("Transfer Characteristics at VBS=0"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax = axes[1,1]
    for vgs_ref, c in zip([0.0,0.3,0.5,0.7,0.9,1.1], plt.cm.plasma(np.linspace(0,1,6))):
        vds_plot = np.linspace(0.001, 1.2, 200)
        xx = np.column_stack([np.full_like(vds_plot, vgs_ref), vds_plot, np.zeros_like(vds_plot)])
        x_t = torch.tensor((xx - X_mean) / X_std, dtype=torch.float32)
        with torch.no_grad(): y_pred = model_cpu(x_t).numpy()
        ax.plot(vds_plot, 10**y_pred, "-", color=c, lw=1.5, label=f"VGS={vgs_ref}V")
    ax.set_xlabel("VDS (V)"); ax.set_ylabel("ID (A)")
    ax.set_title("Output Characteristics at VBS=0"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax = axes[1,2]
    for vbs_ref in [0.0, -0.1, -0.3, -0.5]:
        vgs_plot = np.linspace(-0.2, 1.2, 200)
        xx = np.column_stack([vgs_plot, np.full_like(vgs_plot, 0.8), np.full_like(vgs_plot, vbs_ref)])
        x_t = torch.tensor((xx - X_mean) / X_std, dtype=torch.float32)
        with torch.no_grad(): y_pred = model_cpu(x_t).numpy()
        ax.plot(vgs_plot, 10**y_pred, "-", lw=1.5, label=f"VBS={vbs_ref}V")
    ax.set_xlabel("VGS (V)"); ax.set_ylabel("ID (A)"); ax.set_yscale("log")
    ax.set_title("Body Effect (VDS=0.8V)"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {PLOT_PATH}"); print(f"Model saved to: {MODEL_PATH}")
    return model_cpu, X_mean, X_std

def generate_verilog(model, X_mean, X_std, output_file=VA_PATH):
    H = 32; s = model.state_dict()
    w = {"b1": s["fc1.bias"].numpy(), "w1": s["fc1.weight"].numpy(),
         "b2": s["fc2.bias"].numpy(), "w2": s["fc2.weight"].numpy(),
         "b3": s["fc3.bias"].numpy(), "w3": s["fc3.weight"].numpy()}
    def fmt(val, d=8):
        return "0.0" if abs(val) < 1e-12 else f"{val:.{d}f}"
    L = []
    L.append("// ============================================================")
    L.append("// BSIM-NN v6: SPICE-Compatible Neural Network MOSFET Model")
    L.append("// 3 inputs -> 32 tanh -> 32 tanh -> 1 output (log10 ID)")
    L.append("// ============================================================")
    L.append("// SPICE convergence features:")
    L.append("//  - Smooth tanh-based input clamping (no if/ternary)")
    L.append("//  - Minimum std 0.05 to avoid division by zero")
    L.append("//  - Gate/body conductance paths for Jacobian stability")
    L.append("//  - Weight-regularized to avoid tanh saturation")
    L.append("//  - Smooth VDS->0 transition")
    L.append("// ============================================================")
    L.append(""); L.append("`include \"disciplines.vams\""); L.append("`include \"constants.vams\"")
    L.append(""); L.append("module bsim_nn_v6(d, g, s, b);")
    L.append("    inout d, g, s, b; electrical d, g, s, b;"); L.append("")
    L.append("    parameter real L = 1e-6 from (0:inf);")
    L.append("    parameter real W = 10e-6 from (0:inf);"); L.append("")
    L.append("    real VGS, VDS, VBS, VGS_n, VDS_n, VBS_n;")
    h1_v = ", ".join([f"h1_{i}" for i in range(H)])
    h2_v = ", ".join([f"h2_{i}" for i in range(H)])
    L.append(f"    real {h1_v};"); L.append(f"    real {h2_v};")
    L.append("    real log10_Id, Ids, vds_smooth;"); L.append("")
    L.append(f"    real mean_vgs = {fmt(X_mean[0])};")
    L.append(f"    real std_vgs  = {fmt(X_std[0])};")
    L.append(f"    real mean_vds = {fmt(X_mean[1])};")
    L.append(f"    real std_vds  = {fmt(X_std[1])};")
    L.append(f"    real mean_vbs = {fmt(X_mean[2])};")
    L.append(f"    real std_vbs  = {fmt(X_std[2])};")
    L.append("    analog begin")
    L.append("        VGS = V(g, s); VDS = V(d, s); VBS = V(b, s);")
    L.append("")
    L.append("        // Smooth clamp to training range via tanh")
    L.append("        VGS = 0.5 + 0.7 * tanh((VGS - 0.5) / 0.7);")
    L.append("        VDS = 0.6 + 0.6 * tanh((VDS - 0.6) / 0.6);")
    L.append("        VBS = -0.25 + 0.25 * tanh((VBS + 0.25) / 0.25);")
    L.append("")
    L.append("        VGS_n = (VGS - mean_vgs) / std_vgs;")
    L.append("        VDS_n = (VDS - mean_vds) / std_vds;")
    L.append("        VBS_n = (VBS - mean_vbs) / std_vbs;")
    L.append("")
    L.append("        // Hidden Layer 1 (tanh)"); L.append("")
    for i in range(H):
        t = [fmt(w["b1"][i])]
        t.append(f'{fmt(w["w1"][i,0])}*VGS_n')
        t.append(f'{fmt(w["w1"][i,1])}*VDS_n')
        t.append(f'{fmt(w["w1"][i,2])}*VBS_n')
        L.append(f'        h1_{i} = tanh({" + ".join(t)});')
    L.append(""); L.append("        // Hidden Layer 2 (tanh)"); L.append("")
    for i in range(H):
        t = [fmt(w["b2"][i])]
        for j in range(H):
            wv = w["w2"][i,j]
            if abs(wv) > 1e-10: t.append(f'{fmt(wv)}*h1_{j}')
        L.append(f'        h2_{i} = tanh({" + ".join(t)});')
    L.append(""); L.append("        // Output Layer (linear)"); L.append("")
    t = [fmt(w["b3"][0])]
    for j in range(H):
        wv = w["w3"][0,j]
        if abs(wv) > 1e-10: t.append(f'{fmt(wv)}*h2_{j}')
    L.append(f'        log10_Id = {" + ".join(t)};')
    L.append(""); L.append("        // Smooth output clamping to [-12, -3]")
    L.append("        log10_Id = -7.5 + 4.5 * tanh((log10_Id + 7.5) / 4.5);")
    L.append(""); L.append("        Ids = pow(10, log10_Id) * (W / 10e-6);")
    L.append(""); L.append("        // Smooth VDS->0 transition")
    L.append("        if (VDS >= 0) vds_smooth = VDS; else vds_smooth = 0;")
    L.append("        Ids = Ids * vds_smooth / (vds_smooth + 1e-4);")
    L.append("")
    L.append("        I(d, s) <+ Ids;")
    L.append("        I(g, s) <+ 1e-14 * V(g, s);")
    L.append("        I(b, s) <+ 1e-14 * V(b, s);")
    L.append("    end"); L.append(""); L.append("endmodule")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"\nVerilog-A generated: {output_file} ({len(L)} lines)")
    for k, arr in w.items():
        val = np.abs(arr).max(); status = "OK" if val < 5 else "WARNING: >5"
        print(f"  {k}: max_abs={val:.2f} [{status}]")
    print("\nCompile: openvaf bsim_nn_v6.va --ngspice -o bsim_nn_v6.osdi")

if __name__ == "__main__":
    model, X_mean, X_std = train_model()
    generate_verilog(model, X_mean, X_std)
    print("\nDone! Next: openvaf bsim_nn_v6.va --ngspice -o bsim_nn_v6.osdi")