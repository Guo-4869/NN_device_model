import numpy as np
import pandas as pd
import subprocess
import tempfile
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def run_bsim_simulation(vgs, vds, w=10e-6, l=1e-6, temp=27):
    """
    使用ngspice调用BSIM模型生成单个数据点
    """
    # 创建临时网表
    netlist = f'''BSIM3v3 MOSFET simulation

.options savecurrents
.options noopiter

M1 d g 0 0 N1 W={w*1e6:.2f}u L={l*1e6:.2f}u
.model N1 NMOS level=8 version=3.3.0
+ tox=7.4e-9 nch=1.7e17 vth0=0.7 u0=0.05
+ nsub=5e16 xj=0.25e-6 ld=0.1e-6
+ theta=0.1 lambda=0.02

Vgs g 0 {vgs}
Vds d 0 {vds}

.op

.control
run
print @m1[id]
.endc

.end
'''
    
    # 写入临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cir', delete=False) as f:
        f.write(netlist)
        temp_file = f.name
    
    try:
        # 运行ngspice
        result = subprocess.run(
            ['ngspice', '-b', temp_file],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # 解析输出获取电流
        output = result.stdout
        match = re.search(r'@m1\[id\]\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', output)
        
        if match:
            ids = float(match.group(1))
            return max(ids, 1e-15)
        else:
            return 1e-15
            
    except Exception as e:
        print(f"Simulation failed for Vgs={vgs}, Vds={vds}: {e}")
        return 1e-15
    finally:
        os.unlink(temp_file)


class BSIMDataGenerator:
    """
    BSIM模型数据生成器
    使用ngspice内置的BSIM3v3模型生成训练数据
    """
    
    def __init__(self):
        # BSIM3v3 典型参数 (0.18um 工艺)
        self.bsim_params = {
            'level': 8,
            'version': '3.3.0',
            'tox': 7.4e-9,      # 栅氧化层厚度
            'nch': 1.7e17,      # 沟道掺杂浓度
            'vth0': 0.7,        # 阈值电压
            'u0': 0.05,         # 迁移率
            'nsub': 5e16,       # 衬底掺杂
            'xj': 0.25e-6,      # 结深
            'ld': 0.1e-6,       # 横向扩散
            'theta': 0.1,       # 迁移率退化
            'lambda': 0.02,     # 沟道长度调制
            'wint': 0.05e-6,    # 宽度修正
            'lint': 0.05e-6,    # 长度修正
        }
    
    def generate_grid_data(self, w=10e-6, l=1e-6, temp=27):
        """
        生成网格数据用于训练和验证
        使用BSIM方程快速计算（避免重复调用spice）
        """
        # 电压范围
        vgs_range = np.linspace(0, 3, 61)
        vds_range = np.linspace(0, 3, 31)
        
        Vgs, Vds = np.meshgrid(vgs_range, vds_range)
        Ids = np.zeros_like(Vgs)
        
        # BSIM3v3 简化计算
        vth = self.bsim_params['vth0']
        
        for i, vds in enumerate(vds_range):
            for j, vgs in enumerate(vgs_range):
                if vgs <= vth:
                    ids = 0
                else:
                    vov = vgs - vth
                    if vds < vov:
                        # 线性区
                        ids = self.bsim_params['u0'] * (w/l) * 1e-3 * (2*vov*vds - vds*vds)
                    else:
                        # 饱和区
                        ids = self.bsim_params['u0'] * (w/l) * 1e-3 * vov*vov
                    
                    # 沟道长度调制
                    ids *= (1 + self.bsim_params['lambda'] * vds)
                    
                    # 亚阈值电流 (简化)
                    if vgs < vth + 0.2:
                        vt = 0.0259
                        ids += 1e-6 * (w/l) * np.exp((vgs - vth) / (1.5 * vt))
                
                Ids[i, j] = max(ids, 1e-12)
        
        return Vgs, Vds, Ids, vgs_range, vds_range
    
    def generate_training_data_lhs(self, n_samples=5000, save_csv=True):
        """
        使用拉丁超立方采样生成训练数据
        """
        print(f"生成 {n_samples} 个BSIM训练样本...")
        
        # 参数范围
        bounds = {
            'vgs': (0, 3),      # 栅电压 (V)
            'vds': (0, 3),      # 漏电压 (V)
            'w': (0.5e-6, 20e-6),   # 宽度 (m)
            'l': (0.18e-6, 10e-6),  # 长度 (m)
            'temp': (25, 125)        # 温度 (C)
        }
        
        # 拉丁超立方采样
        samples = np.zeros((n_samples, len(bounds)))
        for i, (key, (low, high)) in enumerate(bounds.items()):
            strata = np.linspace(low, high, n_samples + 1)
            points = np.random.uniform(strata[:-1], strata[1:])
            np.random.shuffle(points)
            samples[:, i] = points
        
        # 提取参数
        vgs_vals = samples[:, 0]
        vds_vals = samples[:, 1]
        w_vals = samples[:, 2]
        l_vals = samples[:, 3]
        temp_vals = samples[:, 4]
        
        # 计算电流
        vth = self.bsim_params['vth0']
        u0 = self.bsim_params['u0']
        lambda_ = self.bsim_params['lambda']
        vt = 0.0259
        
        Ids = np.zeros(n_samples)
        
        for i in range(n_samples):
            vgs = vgs_vals[i]
            vds = vds_vals[i]
            w = w_vals[i]
            l = l_vals[i]
            temp = temp_vals[i]
            
            # 温度对阈值电压的影响
            vth_temp = vth - 0.002 * (temp - 27)
            
            # 温度对迁移率的影响
            u0_temp = u0 * (300 / (temp + 273)) ** 1.5
            
            if vgs <= vth_temp:
                ids = 0
            else:
                vov = vgs - vth_temp
                beta = u0_temp * 1e-3 * (w / l)
                
                if vds < vov:
                    ids = beta * (2 * vov * vds - vds * vds)
                else:
                    ids = beta * vov * vov
                
                # 沟道长度调制
                ids *= (1 + lambda_ * vds)
                
                # 亚阈值电流
                if vgs < vth_temp + 0.3:
                    ids += 1e-6 * (w/l) * np.exp((vgs - vth_temp) / (1.5 * vt))
            
            Ids[i] = max(ids, 1e-12)
        
        # 创建DataFrame
        data = pd.DataFrame({
            'Vgs': vgs_vals,
            'Vds': vds_vals,
            'W': w_vals,
            'L': l_vals,
            'Temp': temp_vals,
            'Ids': Ids
        })
        
        # 添加预处理后的值
        data['logIds'] = np.log10(data['Ids'] + 1e-15)
        data['sqrtIds'] = np.sqrt(data['Ids'] + 1e-15)
        
        # 添加长宽比
        data['W_L_ratio'] = data['W'] / data['L']
        
        if save_csv:
            data.to_csv('bsim_training_data.csv', index=False)
            print(f"数据已保存到 bsim_training_data.csv")
        
        print(f"Ids范围: {Ids.min():.2e} - {Ids.max():.2e} A")
        
        return data


def ids_preprocess_advanced(ids, method='hybrid'):
    """
    高级Ids预处理
    method: 'log', 'sqrt', 'hybrid'
    """
    if method == 'log':
        return np.log10(ids + 1e-15)
    elif method == 'sqrt':
        return np.sqrt(ids + 1e-15)
    else:  # hybrid - 参考论文方法
        ids_scaled = ids * 1e5
        ids_log = np.log10(ids + 1e-15) + 15
        ids_linear = ids_scaled
        
        log_ids = np.log10(ids + 1e-15)
        transition = 1 / (1 + np.exp(-(log_ids + 8)))
        
        return transition * ids_linear + (1 - transition) * ids_log


def vds_preprocess_advanced(vds, vdd=3.0):
    """Vds预处理"""
    return np.power(vds / vdd, 0.8) * vdd


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # 生成数据
    generator = BSIMDataGenerator()
    
    # 生成网格数据用于可视化
    print("生成BSIM网格数据...")
    Vgs, Vds, Ids, vgs_range, vds_range = generator.generate_grid_data(w=10e-6, l=1e-6)
    
    # 生成训练数据
    data = generator.generate_training_data_lhs(n_samples=10000)
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 输出特性曲线
    ax1 = axes[0, 0]
    vgs_samples = [1.0, 1.5, 2.0, 2.5]
    for vg in vgs_samples:
        idx = np.argmin(np.abs(vgs_range - vg))
        ax1.plot(vds_range, Ids[:, idx], label=f'Vgs={vg}V')
    ax1.set_xlabel('Vds (V)')
    ax1.set_ylabel('Ids (A)')
    ax1.set_title('BSIM Output Characteristics')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 转移特性
    ax2 = axes[0, 1]
    vds_samples = [0.1, 0.5, 1.0, 2.0]
    for vd in vds_samples:
        idx = np.argmin(np.abs(vds_range - vd))
        ax2.plot(vgs_range, Ids[idx, :], label=f'Vds={vd}V')
    ax2.set_xlabel('Vgs (V)')
    ax2.set_ylabel('Ids (A)')
    ax2.set_title('BSIM Transfer Characteristics')
    ax2.legend()
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # 训练数据分布
    ax3 = axes[1, 0]
    ax3.scatter(data['Vgs'], data['Ids'], alpha=0.3, s=1)
    ax3.set_xlabel('Vgs (V)')
    ax3.set_ylabel('Ids (A)')
    ax3.set_title('LHS Training Data')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # logIds分布
    ax4 = axes[1, 1]
    ax4.hist(data['logIds'], bins=50, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('log10(Ids)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Ids Distribution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bsim_data_visualization.png', dpi=150)
    plt.show()
    
    print("\n数据统计:")
    print(f"样本数: {len(data)}")
    print(f"Vgs范围: {data['Vgs'].min():.2f} - {data['Vgs'].max():.2f} V")
    print(f"Vds范围: {data['Vds'].min():.2f} - {data['Vds'].max():.2f} V")
    print(f"Ids范围: {data['Ids'].min():.2e} - {data['Ids'].max():.2e} A")