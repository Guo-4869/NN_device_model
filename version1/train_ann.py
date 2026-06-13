import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

def generate_bsim_data(n_samples=20000):
    """生成BSIM训练数据 - 覆盖正常工作区域"""
    np.random.seed(42)
    
    # 重点采样 Vgs 从 0.5 到 2.5V（亚阈值到饱和区）
    vgs = np.random.uniform(0, 3, n_samples)
    vds = np.random.uniform(0, 3, n_samples)
    
    # BSIM模型参数
    vth0 = 0.7
    beta0 = 1e-3
    
    ids = np.zeros(n_samples)
    for i in range(n_samples):
        if vgs[i] <= vth0:
            # 亚阈值区：指数增长
            ids[i] = 1e-9 * np.exp((vgs[i] - vth0) / 0.1)
        elif vds[i] < (vgs[i] - vth0):
            # 线性区
            ids[i] = beta0 * (2*(vgs[i]-vth0)*vds[i] - vds[i]*vds[i])
        else:
            # 饱和区
            ids[i] = beta0 * (vgs[i]-vth0)*(vgs[i]-vth0)
    
    # 取 log10
    log_ids = np.log10(ids + 1e-15)
    
    return vgs, vds, log_ids


def train():
    print("="*60)
    print("训练 ANN MOSFET 模型")
    print("="*60)
    
    print("\n1. 生成训练数据...")
    Vgs, Vds, logIds = generate_bsim_data(20000)
    
    X = np.column_stack([Vgs, Vds])
    y = logIds.reshape(-1, 1)
    
    # 归一化
    print("2. 数据归一化...")
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
    
    # 转换为Tensor
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)
    
    # 定义网络（2-8-1）
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2, 8)
            self.fc2 = nn.Linear(8, 1)
            
            # 初始化权重（使用Xavier初始化）
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.zeros_(self.fc1.bias)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)
        
        def forward(self, x):
            x = torch.tanh(self.fc1(x))
            x = self.fc2(x)
            return x
    
    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    # 训练
    print("3. 训练神经网络...")
    losses = []
    for epoch in range(3000):
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % 500 == 0:
            with torch.no_grad():
                test_loss = criterion(model(X_test_t), y_test_t)
            print(f"   Epoch {epoch:4d}, Loss: {loss.item():.6f}, Test Loss: {test_loss.item():.6f}")
    
    # 评估
    print("\n4. 评估模型...")
    with torch.no_grad():
        pred_train = model(X_train_t).numpy()
        pred_test = model(X_test_t).numpy()
    
    pred_train = scaler_y.inverse_transform(pred_train)
    pred_test = scaler_y.inverse_transform(pred_test)
    y_train_orig = scaler_y.inverse_transform(y_train)
    y_test_orig = scaler_y.inverse_transform(y_test)
    
    train_rmse = np.sqrt(np.mean((pred_train - y_train_orig)**2))
    test_rmse = np.sqrt(np.mean((pred_test - y_test_orig)**2))
    print(f"   训练 RMSE: {train_rmse:.6f}")
    print(f"   测试 RMSE: {test_rmse:.6f}")
    
    # 可视化
    print("\n5. 生成可视化...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 损失曲线
    axes[0].plot(losses)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].set_yscale('log')
    axes[0].grid(True)
    
    # 预测 vs 真实
    axes[1].scatter(y_test_orig, pred_test, alpha=0.3, s=1)
    min_val = min(y_test_orig.min(), pred_test.min())
    max_val = max(y_test_orig.max(), pred_test.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--')
    axes[1].set_xlabel('True log10(Ids)')
    axes[1].set_ylabel('Predicted log10(Ids)')
    axes[1].set_title(f'Test Set (RMSE={test_rmse:.4f})')
    axes[1].grid(True)
    
    # 输出特性曲线对比
    axes[2].set_title('Output Characteristics (Vgs=2V)')
    vgs_test = 2.0
    vds_test = np.linspace(0, 3, 50)
    ids_true = []
    ids_pred = []
    
    for vds in vds_test:
        # 真实值
        if vgs_test <= 0.7:
            ids_true.append(1e-9)
        elif vds < (vgs_test - 0.7):
            ids_true.append(1e-3 * (2*(vgs_test-0.7)*vds - vds*vds))
        else:
            ids_true.append(1e-3 * (vgs_test-0.7)*(vgs_test-0.7))
        
        # 预测值（注意：需要detach）
        X_test_point = scaler_X.transform([[vgs_test, vds]])
        X_test_tensor = torch.FloatTensor(X_test_point)
        with torch.no_grad():
            pred = model(X_test_tensor).numpy()
        pred_ids = 10 ** scaler_y.inverse_transform(pred)[0, 0]
        ids_pred.append(pred_ids)
    
    axes[2].plot(vds_test, ids_true, 'b-', label='True BSIM')
    axes[2].plot(vds_test, ids_pred, 'r--', label='ANN Prediction')
    axes[2].set_xlabel('Vds (V)')
    axes[2].set_ylabel('Ids (A)')
    axes[2].legend()
    axes[2].set_yscale('log')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150)
    plt.show()
    
    # 保存模型
    print("\n6. 保存模型...")
    w1 = model.fc1.weight.detach().numpy()
    b1 = model.fc1.bias.detach().numpy()
    w2 = model.fc2.weight.detach().numpy()
    b2 = model.fc2.bias.detach().numpy()
    
    joblib.dump({
        'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2,
        'scaler_X': scaler_X, 'scaler_y': scaler_y,
    }, 'ann_bsim_model_fixed.pkl')
    
    print("   模型已保存到 ann_bsim_model_fixed.pkl")
    
    # 打印权重用于检查
    print("\n训练后的权重范围:")
    print(f"  w1: [{w1.min():.4f}, {w1.max():.4f}]")
    print(f"  b1: [{b1.min():.4f}, {b1.max():.4f}]")
    print(f"  w2: [{w2.min():.4f}, {w2.max():.4f}]")
    print(f"  b2: {b2[0]:.4f}")
    
    # 打印示例预测
    print("\n示例预测 (Vgs=2.0V, Vds=2.0V):")
    X_test_point = scaler_X.transform([[2.0, 2.0]])
    X_test_tensor = torch.FloatTensor(X_test_point)
    with torch.no_grad():
        pred_log = model(X_test_tensor).numpy()
    pred_ids = 10 ** scaler_y.inverse_transform(pred_log)[0, 0]
    print(f"  预测 Ids = {pred_ids:.6f} A")
    
    # 真实值
    if 2.0 <= 0.7:
        true_ids = 1e-9
    elif 2.0 < (2.0 - 0.7):
        true_ids = 1e-3 * (2*(2.0-0.7)*2.0 - 2.0*2.0)
    else:
        true_ids = 1e-3 * (2.0-0.7)*(2.0-0.7)
    print(f"  真实 Ids = {true_ids:.6f} A")
    
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    
    return model, scaler_X, scaler_y


if __name__ == "__main__":
    train()