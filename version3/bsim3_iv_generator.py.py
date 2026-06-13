# bsim3_iv_generator.py
"""
BSIM3v3-based MOSFET IV Characteristic Generator
Based on BSIM3v3.3 Manual - Chapter 2 & 3

Generates realistic IV characteristics including:
- Transfer characteristics (ID-VGS)
- Output characteristics (ID-VDS)  
- Subthreshold characteristics
- Channel length modulation effects
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class BSIM3Params:
    """BSIM3v3模型参数"""
    # 工艺参数
    tox: float = 2e-9          # 栅氧化层厚度 (m)
    xj: float = 1.5e-7         # 结深 (m)
    nch: float = 1.7e23        # 沟道掺杂浓度 (m^-3)
    nsub: float = 6e22         # 衬底掺杂浓度 (m^-3)
    
    # 阈值电压参数
    vth0: float = 0.7          # 零偏阈值电压 (V)
    k1: float = 0.5            # 体效应系数1 (V^0.5)
    k2: float = 0.1            # 体效应系数2
    nlx: float = 0.1           # 横向非均匀掺杂系数
    
    # 迁移率参数
    u0: float = 0.05           # 低场迁移率 (m^2/V·s)
    ua: float = 0.2            # 垂直场迁移率退化系数 (m/V)
    ub: float = 0.05           # 垂直场迁移率退化系数2 (m^2/V^2)
    uc: float = 0.05           # 体偏置迁移率系数
    
    # 饱和速度参数
    vsat: float = 1e5          # 饱和速度 (m/s)
    
    # 沟道长度调制参数
    pclm: float = 1.0          # CLM参数
    
    # 亚阈值参数
    nfactor: float = 1.2       # 亚阈值斜率因子
    voff: float = -0.1         # 亚阈值偏移电压 (V)
    
    # 寄生电阻
    rdsw: float = 200          # 源漏电阻 (Ω·μm)
    
    # 温度
    temp: float = 300          # 温度 (K)
    
    @property
    def vt(self) -> float:
        """热电压"""
        return 8.617e-5 * self.temp
    
    @property
    def cox(self) -> float:
        """栅氧化层电容 (F/m^2)"""
        eps_ox = 3.9 * 8.854e-12
        return eps_ox / self.tox


class BSIM3IVModel:
    """BSIM3v3 IV特性模型实现"""
    
    def __init__(self, params: BSIM3Params = None):
        self.params = params or BSIM3Params()
        self._init_derived_params()
    
    def _init_derived_params(self):
        """初始化派生参数"""
        # 热电压
        self.vt = self.params.vt
        # 栅氧化层电容
        self.cox = self.params.cox
        # 内置电势
        self.ni = 1.45e16  # 本征载流子浓度 (m^-3)
        self.phis = 2 * self.vt * np.log(self.params.nch / self.ni)
        
    def _calc_vth(self, vgs: float, vds: float, vbs: float, l: float, w: float) -> float:
        """
        计算阈值电压 (BSIM3v3 Eq. 2.1.25)
        包含短沟道效应、窄沟道效应、DIBL效应
        """
        # 基础阈值电压
        vth_long = self.params.vth0 + self.params.k1 * (np.sqrt(self.phis - vbs) - np.sqrt(self.phis))
        vth_long -= self.params.k2 * vbs
        
        # 有效栅压
        vgst = vgs - vth_long
        
        # 特征长度 (短沟道效应)
        xdep = np.sqrt(2 * 11.7 * 8.854e-12 * (self.phis - vbs) / (1.6e-19 * self.params.nch))
        lt = np.sqrt(11.7 * 8.854e-12 * xdep / self.cox)
        
        # 短沟道效应导致的阈值电压降低
        theta = 0.5
        dvth_sc = theta * (2 * (0.8 - self.phis) + vds) * (np.exp(-l/lt) + 2*np.exp(-2*l/lt))
        
        # DIBL效应
        if vgst > 0:
            eta = 0.1  # DIBL系数
            dvth_dibl = eta * vds
        else:
            dvth_dibl = 0
        
        # 最终阈值电压
        vth = vth_long - dvth_sc - dvth_dibl
        
        return max(vth, 0.1)
    
    def _calc_mobility(self, vgst: float, vbs: float) -> float:
        """
        计算有效迁移率 (BSIM3v3 Eq. 2.2.4)
        """
        if vgst <= 0:
            return self.params.u0
        
        # 垂直场
        eeff = (vgst + 2 * self.vt) / (3 * self.params.tox)
        
        # 迁移率退化
        ueff = self.params.u0 / (1 + self.params.ua * eeff + self.params.ub * eeff**2)
        ueff /= (1 + self.params.uc * vbs)
        
        return max(ueff, 0.01)
    
    def _calc_vdsat(self, vgst: float, vth: float, l: float) -> float:
        """
        计算饱和电压 (BSIM3v3 Eq. 2.5.6)
        """
        if vgst <= 0:
            return 0
        
        # 饱和电场
        esat = 2 * self.params.vsat / self._calc_mobility(vgst, 0)
        
        # 体效应系数 (简化)
        abulk = 1.0
        
        # 饱和电压
        vdsat = esat * l * vgst / (abulk * esat * l + vgst)
        
        return max(vdsat, 0)
    
    def _calc_linear_current(self, vgs: float, vds: float, vth: float, 
                               vgst: float, l: float, w: float) -> float:
        """
        计算线性区电流 (BSIM3v3 Eq. 2.5.4)
        """
        if vgst <= 0 or vds <= 0:
            return 0
        
        ueff = self._calc_mobility(vgst, 0)
        esat = 2 * self.params.vsat / ueff
        abulk = 1.0
        
        # 宽长比
        beta = ueff * self.cox * (w / l)
        
        # 线性区电流
        ids_lin = beta * (vgst - abulk * vds/2) * vds / (1 + vds/(esat*l))
        
        return max(ids_lin, 0)
    
    def _calc_saturation_current(self, vgs: float, vds: float, vth: float,
                                    vgst: float, vdsat: float, l: float, w: float) -> float:
        """
        计算饱和区电流 (BSIM3v3 Eq. 2.6.11)
        包含沟道长度调制效应
        """
        if vgst <= 0 or vdsat <= 0:
            return 0
        
        ueff = self._calc_mobility(vgst, 0)
        
        # 饱和区基础电流 (vds = vdsat时)
        beta = ueff * self.cox * (w / l)
        idsat = beta * (vgst * vdsat - vdsat**2/2) / (1 + vdsat/(2*self.params.vsat*l/ueff))
        
        # 沟道长度调制效应
        if vds > vdsat:
            clm_factor = 1 + self.params.pclm * (vds - vdsat)
        else:
            clm_factor = 1
        
        ids = idsat * clm_factor
        
        return max(ids, 0)
    
    def _calc_subthreshold_current(self, vgs: float, vds: float, vth: float, 
                                     l: float, w: float) -> float:
        """
        计算亚阈值电流 (BSIM3v3 Eq. 2.7.1)
        """
        # 亚阈值斜率
        n = self.params.nfactor
        
        # 亚阈值电流
        ids_sub = 1e-6 * (w / l) * np.exp((vgs - vth - self.params.voff) / (n * self.vt))
        
        # 考虑VDS影响
        ids_sub *= (1 - np.exp(-vds / self.vt))
        
        return max(ids_sub, 1e-15)
    
    def calc_ids(self, vgs: float, vds: float, vbs: float = 0, 
                 l: float = 1e-6, w: float = 10e-6) -> float:
        """
        计算漏极电流
        统一模型：结合线性区、饱和区和亚阈值区 (BSIM3v3 Eq. 3.6.1)
        """
        # 阈值电压
        vth = self._calc_vth(vgs, vds, vbs, l, w)
        
        # 有效栅压
        vgst = vgs - vth
        
        # 饱和电压
        vdsat = self._calc_vdsat(vgst, vth, l)
        
        # 计算各区域电流
        ids_lin = self._calc_linear_current(vgs, vds, vth, vgst, l, w)
        ids_sat = self._calc_saturation_current(vgs, vds, vth, vgst, vdsat, l, w)
        ids_sub = self._calc_subthreshold_current(vgs, vds, vth, l, w)
        
        # 平滑过渡 (使用双曲正切)
        if vgst > 0:
            # 强反型区 - 使用平滑过渡连接线性和饱和区
            alpha = 5.0 / vdsat if vdsat > 0 else 1
            transition = 0.5 * (1 + np.tanh(alpha * (vds - vdsat)))
            ids_inv = ids_lin * (1 - transition) + ids_sat * transition
        else:
            ids_inv = 0
        
        # 总电流 = 亚阈值电流 + 反型层电流
        ids = ids_sub + ids_inv
        
        # 寄生电阻效应 (简化)
        if ids > 0 and vds > 0:
            rd = self.params.rdsw / (w * 1e6)  # 电阻值
            ids = ids / (1 + rd * ids / vds)
        
        return max(ids, 1e-15)


class IVDataGenerator:
    """IV数据生成和可视化"""
    
    def __init__(self, model: BSIM3IVModel):
        self.model = model
        
    def generate_transfer_data(self, 
                                vgs_range: Tuple[float, float, int] = (0, 1.2, 100),
                                vds: float = 0.05,
                                l: float = 1e-6,
                                w: float = 10e-6) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成转移特性数据 (ID-VGS)
        """
        vgs = np.linspace(vgs_range[0], vgs_range[1], vgs_range[2])
        ids = np.array([self.model.calc_ids(vg, vds, 0, l, w) for vg in vgs])
        
        return vgs, ids
    
    def generate_output_data(self,
                              vds_range: Tuple[float, float, int] = (0, 1.2, 100),
                              vgs_list: list = [0.4, 0.6, 0.8, 1.0, 1.2],
                              l: float = 1e-6,
                              w: float = 10e-6) -> dict:
        """
        生成输出特性数据 (ID-VDS) 在不同VGS下
        """
        vds = np.linspace(vds_range[0], vds_range[1], vds_range[2])
        results = {}
        
        for vgs in vgs_list:
            ids = np.array([self.model.calc_ids(vgs, vd, 0, l, w) for vd in vds])
            results[vgs] = (vds, ids)
        
        return results
    
    def generate_subthreshold_data(self,
                                    vgs_range: Tuple[float, float, int] = (0, 0.8, 100),
                                    vds_list: list = [0.05, 0.5, 1.0],
                                    l: float = 1e-6,
                                    w: float = 10e-6) -> dict:
        """
        生成亚阈值特性数据
        """
        vgs = np.linspace(vgs_range[0], vgs_range[1], vgs_range[2])
        results = {}
        
        for vds in vds_list:
            ids = np.array([self.model.calc_ids(vg, vds, 0, l, w) for vg in vgs])
            results[vds] = (vgs, ids)
        
        return results
    
    def plot_iv_curves(self, 
                       l: float = 1e-6, 
                       w: float = 10e-6,
                       save_path: Optional[str] = None):
        """
        绘制完整的IV特性曲线
        """
        fig = plt.figure(figsize=(16, 12))
        
        # 1. 转移特性 (线性坐标)
        ax1 = fig.add_subplot(2, 3, 1)
        for vds in [0.05, 0.2, 0.5, 1.0]:
            vgs, ids = self.generate_transfer_data(vds=vds, l=l, w=w)
            ax1.plot(vgs, ids*1e3, linewidth=2, label=f'VDS={vds}V')
        ax1.set_xlabel('VGS (V)', fontsize=12)
        ax1.set_ylabel('ID (mA)', fontsize=12)
        ax1.set_title('Transfer Characteristics (Linear)', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 转移特性 (对数坐标)
        ax2 = fig.add_subplot(2, 3, 2)
        for vds in [0.05, 0.2, 0.5, 1.0]:
            vgs, ids = self.generate_transfer_data(vds=vds, l=l, w=w)
            ax2.semilogy(vgs, ids*1e6, linewidth=2, label=f'VDS={vds}V')
        ax2.set_xlabel('VGS (V)', fontsize=12)
        ax2.set_ylabel('ID (μA)', fontsize=12)
        ax2.set_title('Transfer Characteristics (Log)', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 输出特性
        ax3 = fig.add_subplot(2, 3, 3)
        output_data = self.generate_output_data(vgs_list=[0.4, 0.6, 0.8, 1.0, 1.2], l=l, w=w)
        for vgs, (vds, ids) in output_data.items():
            ax3.plot(vds, ids*1e3, linewidth=2, label=f'VGS={vgs}V')
        ax3.set_xlabel('VDS (V)', fontsize=12)
        ax3.set_ylabel('ID (mA)', fontsize=12)
        ax3.set_title('Output Characteristics', fontsize=14)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 不同L的转移特性
        ax4 = fig.add_subplot(2, 3, 4)
        for l_val in [0.5e-6, 1e-6, 2e-6]:
            vgs, ids = self.generate_transfer_data(vds=0.05, l=l_val, w=w)
            ax4.semilogy(vgs, ids*1e6, linewidth=2, label=f'L={l_val*1e6:.1f}μm')
        ax4.set_xlabel('VGS (V)', fontsize=12)
        ax4.set_ylabel('ID (μA)', fontsize=12)
        ax4.set_title('Transfer Characteristics (Different L)', fontsize=14)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 不同W的转移特性
        ax5 = fig.add_subplot(2, 3, 5)
        for w_val in [5e-6, 10e-6, 20e-6]:
            vgs, ids = self.generate_transfer_data(vds=0.05, l=l, w=w_val)
            ax5.semilogy(vgs, ids*1e6, linewidth=2, label=f'W={w_val*1e6:.0f}μm')
        ax5.set_xlabel('VGS (V)', fontsize=12)
        ax5.set_ylabel('ID (μA)', fontsize=12)
        ax5.set_title('Transfer Characteristics (Different W)', fontsize=14)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. 跨导 (gm)
        ax6 = fig.add_subplot(2, 3, 6)
        vgs, ids = self.generate_transfer_data(vds=0.05, l=l, w=w)
        gm = np.gradient(ids, vgs)
        ax6.plot(vgs, gm*1e3, 'b-', linewidth=2)
        ax6.set_xlabel('VGS (V)', fontsize=12)
        ax6.set_ylabel('gm (mS)', fontsize=12)
        ax6.set_title('Transconductance (gm)', fontsize=14)
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def plot_3d_surface(self, l: float = 1e-6, w: float = 10e-6, 
                         save_path: Optional[str] = None):
        """
        绘制3D表面图
        """
        vgs = np.linspace(0, 1.2, 50)
        vds = np.linspace(0, 1.2, 50)
        VGS, VDS = np.meshgrid(vgs, vds)
        
        ID = np.zeros_like(VGS)
        for i in range(len(vds)):
            for j in range(len(vgs)):
                ID[i, j] = self.model.calc_ids(vgs[j], vds[i], 0, l, w)
        
        fig = plt.figure(figsize=(14, 6))
        
        # 线性尺度
        ax1 = fig.add_subplot(121, projection='3d')
        surf1 = ax1.plot_surface(VGS, VDS, ID*1e3, cmap='viridis', linewidth=0, antialiased=True)
        ax1.set_xlabel('VGS (V)')
        ax1.set_ylabel('VDS (V)')
        ax1.set_zlabel('ID (mA)')
        ax1.set_title('ID-VGS-VDS (Linear)')
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
        
        # 对数尺度
        ax2 = fig.add_subplot(122, projection='3d')
        ID_log = np.log10(ID + 1e-12)
        surf2 = ax2.plot_surface(VGS, VDS, ID_log, cmap='plasma', linewidth=0, antialiased=True)
        ax2.set_xlabel('VGS (V)')
        ax2.set_ylabel('VDS (V)')
        ax2.set_zlabel('log10(ID)')
        ax2.set_title('ID-VGS-VDS (Log Scale)')
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def generate_training_data(self, 
                                l: float = 1e-6, 
                                w: float = 10e-6,
                                vgs_points: int = 50,
                                vds_points: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        生成用于神经网络训练的数据
        返回: (VGS, VDS, ID)
        """
        vgs = np.linspace(0, 1.2, vgs_points)
        vds = np.linspace(0, 1.2, vds_points)
        
        vgs_mesh, vds_mesh = np.meshgrid(vgs, vds)
        id_mesh = np.zeros_like(vgs_mesh)
        
        print(f"Generating {vgs_points * vds_points} data points...")
        
        for i in range(vds_points):
            for j in range(vgs_points):
                id_mesh[i, j] = self.model.calc_ids(vgs[j], vds[i], 0, l, w)
        
        return vgs_mesh, vds_mesh, id_mesh


def main():
    """主函数"""
    print("="*60)
    print("BSIM3v3 MOSFET IV Characteristic Generator")
    print("Based on BSIM3v3.3 Manual")
    print("="*60)
    
    # 创建模型
    params = BSIM3Params()
    model = BSIM3IVModel(params)
    generator = IVDataGenerator(model)
    
    # 器件尺寸
    l = 1e-6      # 1μm沟道长度
    w = 10e-6     # 10μm沟道宽度
    
    print(f"\nDevice dimensions: L = {l*1e6:.1f}μm, W = {w*1e6:.0f}μm")
    print(f"Gate oxide thickness: {params.tox*1e9:.1f}nm")
    print(f"Threshold voltage: {params.vth0:.2f}V")
    
    # 绘制IV特性曲线
    print("\nGenerating IV characteristics...")
    generator.plot_iv_curves(l=l, w=w, save_path="bsim3_iv_curves.png")
    
    # 绘制3D表面图
    print("Generating 3D surface plot...")
    generator.plot_3d_surface(l=l, w=w, save_path="bsim3_3d_surface.png")
    
    # 生成训练数据
    print("\nGenerating training data...")
    vgs, vds, id_data = generator.generate_training_data(l=l, w=w)
    
    # 保存数据到CSV
    import pandas as pd
    vgs_flat = vgs.flatten()
    vds_flat = vds.flatten()
    id_flat = id_data.flatten()
    
    df = pd.DataFrame({
        'VGS': vgs_flat,
        'VDS': vds_flat,
        'ID': id_flat
    })
    df.to_csv('bsim3_training_data.csv', index=False)
    print(f"Training data saved to 'bsim3_training_data.csv'")
    print(f"Total samples: {len(df)}")
    
    # 打印统计信息
    print("\n" + "="*60)
    print("Data Statistics")
    print("="*60)
    print(f"VGS range: [{df['VGS'].min():.2f}, {df['VGS'].max():.2f}] V")
    print(f"VDS range: [{df['VDS'].min():.2f}, {df['VDS'].max():.2f}] V")
    print(f"ID range: [{df['ID'].min():.2e}, {df['ID'].max():.2e}] A")
    
    print("\nDone!")


if __name__ == "__main__":
    main()