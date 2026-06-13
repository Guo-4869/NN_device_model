import numpy as np
import joblib


class VerilogAExporter:
    """
    从训练好的模型生成 Verilog-A 代码
    只使用 Vgs 和 Vds 两个输入
    """
    
    def __init__(self, model_path='ann_bsim_model_fixed.pkl'):
        self.has_trained_model = False
        self.n_hidden = 10
        
        try:
            # 加载训练好的模型
            checkpoint = joblib.load(model_path)
            
            # 提取权重（直接从 pkl 文件）
            self.w1 = checkpoint['w1']        # [hidden, 2]
            self.b1 = checkpoint['b1']        # [hidden]
            self.w2 = checkpoint['w2'][0]     # [hidden] (取第一行)
            self.b2 = checkpoint['b2'][0]     # scalar
            self.scaler_X = checkpoint['scaler_X']
            self.scaler_y = checkpoint['scaler_y']
            
            self.n_hidden = self.w1.shape[0]
            self.has_trained_model = True
            
            # 提取归一化参数
            self.mean_vgs = self.scaler_X.mean_[0]
            self.scale_vgs = self.scaler_X.scale_[0]
            self.mean_vds = self.scaler_X.mean_[1]
            self.scale_vds = self.scaler_X.scale_[1]
            self.mean_logids = self.scaler_y.mean_[0]
            self.scale_logids = self.scaler_y.scale_[0]
            
            print(f"加载训练模型成功")
            print(f"  - 隐藏层神经元: {self.n_hidden}")
            print(f"  - 输入特征: Vgs, Vds")
            print(f"  - Vgs 归一化: mean={self.mean_vgs:.3f}, scale={self.scale_vgs:.3f}")
            print(f"  - Vds 归一化: mean={self.mean_vds:.3f}, scale={self.scale_vds:.3f}")
            
        except FileNotFoundError:
            print(f"未找到训练模型 {model_path}，使用示例权重")
            self._create_example_weights()
        except Exception as e:
            print(f"加载模型出错: {e}")
            self._create_example_weights()
    
    def _create_example_weights(self):
        """创建示例权重（可工作的5神经元示例）"""
        self.n_hidden = 5
        self.w1 = np.array([
            [0.5234, 0.2134],
            [-0.4123, 0.8234],
            [0.1234, -0.6123],
            [-0.2345, 0.3456],
            [0.3456, -0.2345]
        ])
        self.b1 = np.array([-0.3124, 0.1234, 0.4567, -0.2345, 0.3456])
        self.w2 = np.array([0.7234, 0.9123, -0.3345, 0.4567, -0.2123])
        self.b2 = 0.0234
        
        self.mean_vgs = 1.5
        self.scale_vgs = 0.9
        self.mean_vds = 1.5
        self.scale_vds = 0.9
        self.mean_logids = -8.5
        self.scale_logids = 3.2
        
        print(f"使用示例权重: {self.n_hidden}个隐藏层神经元")
    
    def generate_verilog_a(self, output_file='ann_bsim.va'):
        """生成 Verilog-A 模型"""
        
        n_hidden = self.n_hidden
        
        # 生成隐藏层表达式
        hidden_exprs = []
        for i in range(n_hidden):
            w_vgs = self.w1[i, 0]
            w_vds = self.w1[i, 1]
            bias = self.b1[i]
            
            # 格式化，保留6位小数
            hidden_exprs.append(
                f"        h{i+1} = tanh({w_vgs:.6f}*Vgs_n + {w_vds:.6f}*Vds_n + {bias:.6f});"
            )
        
        # 生成输出层表达式
        output_terms = []
        for i in range(n_hidden):
            w_val = self.w2[i]
            if abs(w_val) > 1e-6:
                output_terms.append(f"{w_val:.6f}*h{i+1}")
        
        output_sum = ' + '.join(output_terms) if output_terms else '0'
        
        # 生成变量声明
        hidden_vars = ', '.join([f'h{i+1}' for i in range(n_hidden)])
        
        # 生成完整 Verilog-A 代码
        va_code = f'''// ANN-based MOSFET Model
// Trained on BSIM4 data
// Generated from PyTorch training
// Hidden layer: {n_hidden} neurons

`include "disciplines.vams"

module ann_bsim(d, g, s);
    inout d, g, s;
    electrical d, g, s;
    
    // Geometry scaling parameters
    parameter real w = 10e-6;
    parameter real l = 1e-6;
    
    real Vgs, Vds, Ids;
    real {hidden_vars};
    
    // Normalized inputs
    real Vgs_n, Vds_n;
    
    // Normalization parameters (from training data)
    real mean_vgs = {self.mean_vgs:.6f};
    real scale_vgs = {self.scale_vgs:.6f};
    real mean_vds = {self.mean_vds:.6f};
    real scale_vds = {self.scale_vds:.6f};
    
    analog begin
        // Get input voltages
        Vgs = V(g, s);
        Vds = V(d, s);
        
        // Normalize inputs
        Vgs_n = (Vgs - mean_vgs) / scale_vgs;
        Vds_n = (Vds - mean_vds) / scale_vds;
        
        // Hidden layer (tanh activation)
'''
        
        # 添加隐藏层
        for expr in hidden_exprs:
            va_code += expr + '\n'
        
        va_code += f'''
        // Output layer (linear)
        Ids = {output_sum} + {self.b2:.6f};
        
        // Apply geometry scaling and convert to linear current
        Ids = pow(10, Ids) * (w / l);
        
        // Physical constraints
        if (Ids < 1e-12) Ids = 1e-12;
        if (Ids > 0.01) Ids = 0.01;
        
        // Output current
        I(d, s) <+ Ids;
        I(g, s) <+ 0;
    end
endmodule
'''
        
        with open(output_file, 'w') as f:
            f.write(va_code)
        
        print(f"\nVerilog-A模型已保存到 {output_file}")
        print(f"  - 隐藏层: {n_hidden} 个神经元")
        print(f"  - 输出公式: Ids = pow(10, {output_sum} + {self.b2:.6f}) * (w/l)")
        
        return va_code


def main():
    print("="*60)
    print("生成 ANN BSIM Verilog-A 模型")
    print("="*60)
    
    exporter = VerilogAExporter('ann_bsim_model_fixed.pkl')
    exporter.generate_verilog_a('ann_bsim.va')
    
    print("\n" + "="*60)
    print("下一步操作:")
    print("="*60)
    print("\n1. 编译 OSDI 模型:")
    print("   openvaf ann_bsim.va")
    print("\n2. 创建测试网表 test.cir:")
    print('''
* Test ANN BSIM Model

.control
pre_osdi ann_bsim.osdi
.endc

.model NMOS1 ann_bsim(w=10e-6 l=1e-6)

N1 d g 0 NMOS1

Vds d 0 0
Vgs g 0 0

.dc Vgs 0 3 0.05 Vds 0 3 0.5

.options savecurrents

.control
run
plot vds#branch
.endc

.end
''')
    print("\n3. 运行仿真:")
    print("   ngspice test.cir")


if __name__ == "__main__":
    main()