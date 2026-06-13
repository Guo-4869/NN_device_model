import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

def generate_bsim_data(n_samples=30000):
    """生成BSIM训练数据 - 分区域采样保证各区域都有足够数据"""
    np.random.seed(42)
    
    # 分区域采样，确保各区域数据均衡
    n_sub = n_samples // 3
    
    # 亚阈值区: Vgs 0-0.7V
    vgs_sub = np.random.uniform(0, 0.7, n_sub)
    vds_sub = np.random.uniform(0, 1, n_sub)
    
    # 线性区: Vgs 0.7-2V, Vds < Vgs-0.7
    vgs_lin = np.random.uniform(0.7, 2.5, n_sub)
    vds_lin = np.random.uniform(0, 1.5, n_sub)
    # 确保 vds < vgs-0.7
    for i in range(n_sub):
        if vds_lin[i] >= vgs_lin[i] - 0.7:
            vds_lin[i] = (vgs_lin[i] - 0.7) * 0.8
    
    # 饱和区: Vgs 0.7-3V, Vds > Vgs-0.7
    vgs_sat = np.random.uniform(0.7, 3, n_sub)
    vds_sat = np.random.uniform(0.5, 3, n_sub)
    for i in range(n_sub):
        if vds_sat[i] <= vgs_sat[i] - 0.7:
            vds_sat[i] = (vgs_sat[i] - 0.7) + 0.2
    
    # 合并数据
    vgs = np.concatenate([vgs_sub, vgs_lin, vgs_sat])
    vds = np.concatenate([vds_sub, vds_lin, vds_sat])
    
    # 打乱顺序
    idx = np.random.permutation(len(vgs))
    vgs = vgs[idx]
    vds = vds[idx]
    
    # BSIM模型参数
    vth0 = 0.7
    beta0 = 1e-3
    
    ids = np.zeros(len(vgs))
    region = np.zeros(len(vgs))  # 0:亚阈值, 1:线性, 2:饱和
    
    for i in range(len(vgs)):
        if vgs[i] <= vth0:
            # 亚阈值区：指数关系
            ids[i] = 1e-9 * np.exp((vgs[i] - vth0) / 0.08)
            region[i] = 0
        elif vds[i] < (vgs[i] - vth0):
            # 线性区
            ids[i] = beta0 * (2*(vgs[i]-vth0)*vds[i] - vds[i]*vds[i])
            region[i] = 1
        else:
            # 饱和区
            ids[i] = beta0 * (vgs[i]-vth0)*(vgs[i]-vth0)
            region[i] = 2
    
    # 取 log10
    log_ids = np.log10(ids + 1e-15)
    
    return vgs, vds, log_ids, region


class PhysicsAwareANN(nn.Module):
    """物理启发的 ANN 网络"""
    def __init__(self, hidden_size=12):
        super().__init__()
        
        # 特征提取后维度: Vgs特征(3) + Vds特征(3) = 6
        self.hidden = nn.Sequential(
            nn.Linear(6, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # 可学习的缩放参数
        self.vgs_scale = nn.Parameter(torch.tensor(0.5))
        self.vds_sat = nn.Parameter(torch.tensor(0.8))
    
    def forward(self, vgs, vds):
        # Vgs 特征提取
        vgs_linear = vgs
        vgs_square = vgs ** 2
        vgs_exp = torch.exp(vgs * self.vgs_scale) - 1.0
        
        # Vds 特征提取
        vds_linear = vds
        vds_square = vds ** 2
        vds_sat = torch.tanh(vds * self.vds_sat)
        
        # 拼接特征
        features = torch.cat([vgs_linear, vgs_square, vgs_exp, 
                              vds_linear, vds_square, vds_sat], dim=1)
        
        return self.hidden(features)


class WeightedMSELoss(nn.Module):
    """
    加权 MSE 损失函数
    给不同电流区域的样本不同的权重，平衡亚阈值区的贡献
    """
    def __init__(self, region_weights={0: 3.0, 1: 1.0, 2: 1.0}):
        super().__init__()
        self.region_weights = region_weights
    
    def forward(self, pred, target, region):
        # 基础 MSE
        mse = (pred - target) ** 2
        
        # 根据区域施加权重
        weights = torch.ones_like(region)
        for r, w in self.region_weights.items():
            weights[region == r] = w
        
        return (mse * weights).mean()


def train():
    print("="*60)
    print("物理启发的 ANN MOSFET 模型训练 (加权损失)")
    print("="*60)
    
    # 生成数据
    print("\n1. 生成训练数据...")
    Vgs, Vds, logIds, region = generate_bsim_data(30000)
    
    X = np.column_stack([Vgs, Vds])
    y = logIds.reshape(-1, 1)
    
    # 统计各区域数据量
    print(f"   亚阈值区: {np.sum(region == 0)} 样本")
    print(f"   线性区: {np.sum(region == 1)} 样本")
    print(f"   饱和区: {np.sum(region == 2)} 样本")
    print(f"   Ids 范围: {10**y.min():.2e} - {10**y.max():.2e} A")
    
    # 归一化输入
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    # 归一化输出 (使用 log scale)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)
    
    # 划分数据集
    n_train = int(0.8 * len(X_scaled))
    X_train = X_scaled[:n_train]
    y_train = y_scaled[:n_train]
    X_test = X_scaled[n_train:]
    y_test = y_scaled[n_train:]
    region_train = region[:n_train]
    region_test = region[n_train:]
    
    # 转换为 Tensor
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)
    region_train_t = torch.LongTensor(region_train)
    region_test_t = torch.LongTensor(region_test)
    
    # 创建模型
    model = PhysicsAwareANN(hidden_size=12)
    
    # 使用加权损失
    criterion = WeightedMSELoss(region_weights={0: 5.0, 1: 1.0, 2: 1.0})
    # 同时使用标准 MSE 用于监控
    criterion_mse = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=300, factor=0.5)
    
    # 训练
    print("\n2. 训练神经网络...")
    losses = []
    losses_mse = []
    test_losses = []
    
    for epoch in range(400):
        optimizer.zero_grad()
        
        # 需要原始输入值（非标准化）用于特征提取
        vgs_t = torch.FloatTensor(X_train[:, 0:1])
        vds_t = torch.FloatTensor(X_train[:, 1:2])
        
        output = model(vgs_t, vds_t)
        
        # 加权损失
        loss = criterion(output, y_train_t, region_train_t)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(loss.item())
        losses_mse.append(criterion_mse(output, y_train_t).item())
        
        # 测试集评估
        with torch.no_grad():
            vgs_test_t = torch.FloatTensor(X_test[:, 0:1])
            vds_test_t = torch.FloatTensor(X_test[:, 1:2])
            test_output = model(vgs_test_t, vds_test_t)
            test_loss = criterion(test_output, y_test_t, region_test_t)
            test_losses.append(test_loss.item())
        
        scheduler.step(test_loss)
        
        if epoch % 10 == 0:
            print(f"   Epoch {epoch:4d}, Weighted Loss: {loss.item():.6f}, "
                  f"MSE: {losses_mse[-1]:.6f}, Test: {test_loss.item():.6f}")
    
    # 评估
    print("\n3. 评估模型...")
    with torch.no_grad():
        vgs_test_t = torch.FloatTensor(X_test[:, 0:1])
        vds_test_t = torch.FloatTensor(X_test[:, 1:2])
        pred_test = model(vgs_test_t, vds_test_t).numpy()
    
    pred_test = scaler_y.inverse_transform(pred_test)
    y_test_orig = scaler_y.inverse_transform(y_test)
    
    # 分区域评估
    test_rmse_total = np.sqrt(np.mean((pred_test - y_test_orig)**2))
    print(f"   总测试 RMSE: {test_rmse_total:.6f}")
    
    for r_name, r_val in [('亚阈值', 0), ('线性', 1), ('饱和', 2)]:
        mask = region_test == r_val
        if np.sum(mask) > 0:
            rmse = np.sqrt(np.mean((pred_test[mask] - y_test_orig[mask])**2))
            print(f"   {r_name}区 RMSE: {rmse:.6f} ({np.sum(mask)} 样本)")
    
    # 可视化
    print("\n4. 生成可视化...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 损失曲线
    axes[0,0].plot(losses, label='Weighted Loss')
    axes[0,0].plot(losses_mse, label='MSE')
    axes[0,0].plot(test_losses, label='Test')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].set_title('Training Loss')
    axes[0,0].set_yscale('log')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # 预测 vs 真实
    axes[0,1].scatter(y_test_orig, pred_test, alpha=0.3, s=1, c=region_test, cmap='viridis')
    min_val = min(y_test_orig.min(), pred_test.min())
    max_val = max(y_test_orig.max(), pred_test.max())
    axes[0,1].plot([min_val, max_val], [min_val, max_val], 'r--')
    axes[0,1].set_xlabel('True log10(Ids)')
    axes[0,1].set_ylabel('Predicted log10(Ids)')
    axes[0,1].set_title(f'Test Set (RMSE={test_rmse_total:.4f})')
    axes[0,1].grid(True)
    
    # 输出特性曲线 (Vgs=2V)
    axes[0,2].set_title('Output Characteristics (Vgs=2V)')
    vgs_fixed = 2.0
    vds_range = np.linspace(0, 3, 50)
    ids_true, ids_pred = [], []
    
    for vds_val in vds_range:
        if vgs_fixed <= 0.7:
            ids_true.append(1e-9)
        elif vds_val < (vgs_fixed - 0.7):
            ids_true.append(1e-3 * (2*(vgs_fixed-0.7)*vds_val - vds_val*vds_val))
        else:
            ids_true.append(1e-3 * (vgs_fixed-0.7)*(vgs_fixed-0.7))
        
        with torch.no_grad():
            pred = model(torch.FloatTensor([[vgs_fixed]]), 
                        torch.FloatTensor([[vds_val]])).numpy()
        pred_ids = 10 ** scaler_y.inverse_transform(pred)[0, 0]
        ids_pred.append(pred_ids)
    
    axes[0,2].plot(vds_range, ids_true, 'b-', label='True')
    axes[0,2].plot(vds_range, ids_pred, 'r--', label='ANN')
    axes[0,2].set_xlabel('Vds (V)')
    axes[0,2].set_ylabel('Ids (A)')
    axes[0,2].legend()
    axes[0,2].set_yscale('log')
    axes[0,2].grid(True)
    
    # 转移特性 (Vds=2V) - 线性坐标
    axes[1,0].set_title('Transfer Characteristics (Vds=2V)')
    vds_fixed = 2.0
    vgs_range = np.linspace(0, 3, 50)
    ids_true, ids_pred = [], []
    
    for vgs_val in vgs_range:
        if vgs_val <= 0.7:
            ids_true.append(1e-9)
        elif vds_fixed < (vgs_val - 0.7):
            ids_true.append(1e-3 * (2*(vgs_val-0.7)*vds_fixed - vds_fixed*vds_fixed))
        else:
            ids_true.append(1e-3 * (vgs_val-0.7)*(vgs_val-0.7))
        
        with torch.no_grad():
            pred = model(torch.FloatTensor([[vgs_val]]), 
                        torch.FloatTensor([[vds_fixed]])).numpy()
        pred_ids = 10 ** scaler_y.inverse_transform(pred)[0, 0]
        ids_pred.append(pred_ids)
    
    axes[1,0].plot(vgs_range, ids_true, 'b-', label='True')
    axes[1,0].plot(vgs_range, ids_pred, 'r--', label='ANN')
    axes[1,0].set_xlabel('Vgs (V)')
    axes[1,0].set_ylabel('Ids (A)')
    axes[1,0].legend()
    axes[1,0].set_yscale('log')
    axes[1,0].grid(True)
    
    # 转移特性 - 对数坐标 (展示亚阈值区)
    axes[1,1].set_title('Subthreshold Region (log scale)')
    vgs_sub = np.linspace(0, 1.2, 50)
    ids_true_sub, ids_pred_sub = [], []
    
    for vgs_val in vgs_sub:
        if vgs_val <= 0.7:
            ids_true_sub.append(1e-9 * np.exp((vgs_val - 0.7) / 0.08))
        else:
            ids_true_sub.append(1e-3 * (vgs_val-0.7)*(vgs_val-0.7))
        
        with torch.no_grad():
            pred = model(torch.FloatTensor([[vgs_val]]), 
                        torch.FloatTensor([[2.0]])).numpy()
        pred_ids = 10 ** scaler_y.inverse_transform(pred)[0, 0]
        ids_pred_sub.append(pred_ids)
    
    axes[1,1].semilogy(vgs_sub, ids_true_sub, 'b-', label='True')
    axes[1,1].semilogy(vgs_sub, ids_pred_sub, 'r--', label='ANN')
    axes[1,1].set_xlabel('Vgs (V)')
    axes[1,1].set_ylabel('Ids (A)')
    axes[1,1].set_title('Subthreshold Region Detail')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    # 权重分布
    axes[1,2].set_title('Weight Distribution')
    # 提取第一层权重
    weights = model.hidden[0].weight.detach().numpy()
    axes[1,2].hist(weights.flatten(), bins=50, alpha=0.7)
    axes[1,2].set_xlabel('Weight Value')
    axes[1,2].set_ylabel('Frequency')
    axes[1,2].grid(True)
    
    plt.tight_layout()
    plt.savefig('weighted_training_results.png', dpi=150)
    plt.show()
    
    # 提取模型参数
    print("\n5. 保存模型...")
    
    w1 = model.hidden[0].weight.detach().numpy()
    b1 = model.hidden[0].bias.detach().numpy()
    w2 = model.hidden[2].weight.detach().numpy()
    b2 = model.hidden[2].bias.detach().numpy()
    w3 = model.hidden[4].weight.detach().numpy()
    b3 = model.hidden[4].bias.detach().numpy()
    
    model_params = {
        'type': 'PhysicsAwareANN',
        'w1': w1, 'b1': b1,
        'w2': w2, 'b2': b2,
        'w3': w3, 'b3': b3,
        'vgs_scale': model.vgs_scale.detach().numpy(),
        'vds_sat': model.vds_sat.detach().numpy(),
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'hidden_size': 12,
        'n_layers': 3
    }
    
    joblib.dump(model_params, 'ann_improved_model.pkl')
    print("   模型已保存到 ann_improved_model.pkl")
    
    # 打印示例预测
    print("\n示例预测:")
    test_points = [(1.0, 2.0), (1.5, 2.0), (2.0, 2.0), (2.5, 2.0)]
    print("   Vgs  Vds    预测 Ids      真实 Ids")
    for vg, vd in test_points:
        with torch.no_grad():
            pred = model(torch.FloatTensor([[vg]]), torch.FloatTensor([[vd]])).numpy()
        pred_ids = 10 ** scaler_y.inverse_transform(pred)[0, 0]
        
        if vg <= 0.7:
            true_ids = 1e-9 * np.exp((vg - 0.7) / 0.08)
        elif vd < (vg - 0.7):
            true_ids = 1e-3 * (2*(vg-0.7)*vd - vd*vd)
        else:
            true_ids = 1e-3 * (vg-0.7)*(vg-0.7)
        
        print(f"   {vg:.1f}  {vd:.1f}   {pred_ids:.4e} A    {true_ids:.4e} A")
    
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    
    return model, scaler_X, scaler_y, model_params


if __name__ == "__main__":
    train()