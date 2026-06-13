import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

def generate_bsim_data(n_samples=20000):
    """生成BSIM训练数据"""
    np.random.seed(42)
    
    # 重点采样 Vgs 从 0 到 3V，Vds 从 0 到 3V
    vgs = np.random.uniform(0, 3, n_samples)
    vds = np.random.uniform(0, 3, n_samples)
    
    # BSIM模型参数
    vth0 = 0.7
    beta0 = 1e-3
    
    ids = np.zeros(n_samples)
    for i in range(n_samples):
        if vgs[i] <= vth0:
            # 亚阈值区：指数关系
            ids[i] = 1e-9 * np.exp((vgs[i] - vth0) / 0.08)
        elif vds[i] < (vgs[i] - vth0):
            # 线性区：Vds 线性，Vgs 平方
            ids[i] = beta0 * (2*(vgs[i]-vth0)*vds[i] - vds[i]*vds[i])
        else:
            # 饱和区：Vds 饱和，Vgs 平方
            ids[i] = beta0 * (vgs[i]-vth0)*(vgs[i]-vth0)
    
    # 取 log10（压缩动态范围）
    log_ids = np.log10(ids + 1e-15)
    
    return vgs, vds, log_ids


class PhysicsAwareANN(nn.Module):
    """
    物理启发的 ANN 网络
    - 对 Vgs 做特征提取：Vgs, Vgs^2, exp(Vgs*scale)
    - 对 Vds 做特征提取：Vds, Vds^2, sat(Vds)
    - 然后拼接送入隐藏层
    """
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
        
        # 可学习的缩放参数（可选）
        self.vgs_scale = nn.Parameter(torch.tensor(1.0))
        self.vds_sat = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, vgs, vds):
        # Vgs 特征提取：线性、平方、指数
        vgs_linear = vgs
        vgs_square = vgs ** 2
        vgs_exp = torch.exp(vgs * self.vgs_scale) - 1.0
        
        # Vds 特征提取：线性、平方、饱和（tanh 模拟）
        vds_linear = vds
        vds_square = vds ** 2
        vds_sat = torch.tanh(vds * self.vds_sat)
        
        # 拼接特征
        features = torch.cat([vgs_linear, vgs_square, vgs_exp, 
                              vds_linear, vds_square, vds_sat], dim=1)
        
        return self.hidden(features)


class SimplePowerLawANN(nn.Module):
    """
    更简单的幂律网络：直接学习 log(Ids) = a0 + a1*log(Vgs-vth) + a2*log(1+Vds) + ...
    但将 Vgs 和 Vds 分开处理
    """
    def __init__(self, hidden_size=8):
        super().__init__()
        
        # Vgs 特征: Vgs, (Vgs-vth)如果>0 else 0, ln(Vgs+0.1)
        # Vds 特征: Vds, ln(1+Vds)
        # 总特征: 5维
        self.fc1 = nn.Linear(5, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        
    def forward(self, vgs, vds, vth=0.7):
        # Vgs 特征
        vgs_eff = torch.relu(vgs - vth)  # 有效栅压
        vgs_log = torch.log(vgs + 0.1)    # 对数变换
        vgs_exp = torch.exp(vgs * 0.5)    # 指数变换
        
        # Vds 特征
        vds_lin = vds
        vds_log = torch.log(1 + vds)
        
        features = torch.cat([vgs.unsqueeze(1), vgs_eff.unsqueeze(1), 
                              vgs_log.unsqueeze(1), vds_lin.unsqueeze(1), 
                              vds_log.unsqueeze(1)], dim=1)
        
        x = torch.tanh(self.fc1(features))
        return self.fc2(x)


def train():
    print("="*60)
    print("物理启发的 ANN MOSFET 模型训练")
    print("="*60)
    
    # 生成数据
    print("\n1. 生成训练数据...")
    Vgs, Vds, logIds = generate_bsim_data(30000)
    
    X = np.column_stack([Vgs, Vds])
    y = logIds.reshape(-1, 1)
    
    # 归一化输出
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # 划分数据集
    n_train = int(0.8 * len(X_scaled))
    X_train = X_scaled[:n_train]
    y_train = y_scaled[:n_train]
    X_test = X_scaled[n_train:]
    y_test = y_scaled[n_train:]
    
    # 转换为 Tensor
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)
    
    # 创建模型（可以选择不同的网络）
    model = PhysicsAwareANN(hidden_size=12)
    # model = SimplePowerLawANN(hidden_size=8)
    
    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, factor=0.5)
    criterion = nn.MSELoss()
    
    # 训练
    print("3. 训练神经网络...")
    losses = []
    test_losses = []
    
    for epoch in range(3000):
        optimizer.zero_grad()
        
        # 需要原始输入值（非标准化）用于特征提取
        vgs_t = torch.FloatTensor(X_train[:, 0:1])
        vds_t = torch.FloatTensor(X_train[:, 1:2])
        
        # 模型前向传播
        output = model(vgs_t, vds_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(loss.item())
        
        # 测试集评估
        with torch.no_grad():
            vgs_test_t = torch.FloatTensor(X_test[:, 0:1])
            vds_test_t = torch.FloatTensor(X_test[:, 1:2])
            test_output = model(vgs_test_t, vds_test_t)
            test_loss = criterion(test_output, y_test_t)
            test_losses.append(test_loss.item())
        
        scheduler.step(test_loss)
        
        if epoch % 500 == 0:
            print(f"   Epoch {epoch:4d}, Loss: {loss.item():.6f}, Test Loss: {test_loss.item():.6f}")
    
    # 评估
    print("\n4. 评估模型...")
    with torch.no_grad():
        vgs_test_t = torch.FloatTensor(X_test[:, 0:1])
        vds_test_t = torch.FloatTensor(X_test[:, 1:2])
        pred_test = model(vgs_test_t, vds_test_t).numpy()
    
    pred_test = scaler_y.inverse_transform(pred_test)
    y_test_orig = scaler_y.inverse_transform(y_test)
    
    test_rmse = np.sqrt(np.mean((pred_test - y_test_orig)**2))
    print(f"   测试 RMSE: {test_rmse:.6f}")
    
    # 可视化
    print("\n5. 生成可视化...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 损失曲线
    axes[0,0].plot(losses, label='Train')
    axes[0,0].plot(test_losses, label='Test')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].set_title('Training Loss')
    axes[0,0].set_yscale('log')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # 预测 vs 真实
    axes[0,1].scatter(y_test_orig, pred_test, alpha=0.3, s=1)
    min_val = min(y_test_orig.min(), pred_test.min())
    max_val = max(y_test_orig.max(), pred_test.max())
    axes[0,1].plot([min_val, max_val], [min_val, max_val], 'r--')
    axes[0,1].set_xlabel('True log10(Ids)')
    axes[0,1].set_ylabel('Predicted log10(Ids)')
    axes[0,1].set_title(f'Test Set (RMSE={test_rmse:.4f})')
    axes[0,1].grid(True)
    
    # 输出特性曲线 (Vgs=2V)
    axes[0,2].set_title('Output Characteristics (Vgs=2V)')
    vgs_fixed = 2.0
    vds_range = np.linspace(0, 3, 50)
    ids_true, ids_pred = [], []
    
    for vds_val in vds_range:
        # 真实值
        if vgs_fixed <= 0.7:
            ids_true.append(1e-9)
        elif vds_val < (vgs_fixed - 0.7):
            ids_true.append(1e-3 * (2*(vgs_fixed-0.7)*vds_val - vds_val*vds_val))
        else:
            ids_true.append(1e-3 * (vgs_fixed-0.7)*(vgs_fixed-0.7))
        
        # 预测值
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
    
    # 转移特性 (Vds=2V)
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
    
    # 族曲线
    axes[1,1].set_title('Output Characteristics Family')
    vgs_list = [1.0, 1.5, 2.0, 2.5]
    colors = ['b', 'g', 'r', 'c']
    vds_plot = np.linspace(0, 3, 50)
    
    for i, vg in enumerate(vgs_list):
        ids_plot = []
        for vd in vds_plot:
            with torch.no_grad():
                pred = model(torch.FloatTensor([[vg]]), 
                            torch.FloatTensor([[vd]])).numpy()
            ids_plot.append(10 ** scaler_y.inverse_transform(pred)[0, 0])
        axes[1,1].plot(vds_plot, ids_plot, f'{colors[i]}-', label=f'Vgs={vg}V')
    axes[1,1].set_xlabel('Vds (V)')
    axes[1,1].set_ylabel('Ids (A)')
    axes[1,1].set_title('ANN Predicted Family')
    axes[1,1].legend()
    axes[1,1].set_yscale('log')
    axes[1,1].grid(True)
    
    # 权重分布
    axes[1,2].set_title('Feature Importance Analysis')
    with torch.no_grad():
        # 分析各特征的影响
        sample_vgs = torch.FloatTensor([[2.0]])
        sample_vds = torch.FloatTensor([[2.0]])
        
        # 提取第一层权重
        if hasattr(model, 'fc1'):
            weights = model.fc1.weight.detach().numpy()
            axes[1,2].hist(weights.flatten(), bins=50, alpha=0.7)
        elif hasattr(model, 'hidden'):
            # PhysicsAwareANN 的结构
            weights = model.hidden[0].weight.detach().numpy()
            axes[1,2].hist(weights.flatten(), bins=50, alpha=0.7)
        axes[1,2].set_xlabel('Weight Value')
        axes[1,2].set_ylabel('Frequency')
        axes[1,2].set_title('Weight Distribution')
        axes[1,2].grid(True)
    
    plt.tight_layout()
    plt.savefig('improved_training_results.png', dpi=150)
    plt.show()
    
    # 提取模型参数
    print("\n6. 提取模型参数...")
    
    if isinstance(model, PhysicsAwareANN):
        # 提取 PhysicsAwareANN 的参数
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
    else:
        # 提取 SimplePowerLawANN 的参数
        w1 = model.fc1.weight.detach().numpy()
        b1 = model.fc1.bias.detach().numpy()
        w2 = model.fc2.weight.detach().numpy()
        b2 = model.fc2.bias.detach().numpy()
        
        model_params = {
            'type': 'SimplePowerLawANN',
            'w1': w1, 'b1': b1,
            'w2': w2, 'b2': b2,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'hidden_size': 8,
            'n_layers': 2
        }
    
    joblib.dump(model_params, 'ann_improved_model.pkl')
    print("   模型已保存到 ann_improved_model.pkl")
    
    # 打印示例预测
    print("\n示例预测 (Vgs=2.0V, Vds=2.0V):")
    with torch.no_grad():
        pred_log = model(torch.FloatTensor([[2.0]]), torch.FloatTensor([[2.0]])).numpy()
    pred_ids = 10 ** scaler_y.inverse_transform(pred_log)[0, 0]
    print(f"  预测 Ids = {pred_ids:.6f} A")
    
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    
    return model, scaler_X, scaler_y, model_params


if __name__ == "__main__":
    train()