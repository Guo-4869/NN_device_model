import numpy as np
import joblib


class ImprovedVerilogAExporter:
    """导出物理启发的 ANN 模型到 Verilog-A"""
    
    def __init__(self, model_path='ann_improved_model.pkl'):
        checkpoint = joblib.load(model_path)
        
        self.model_type = checkpoint['type']
        self.scaler_X = checkpoint['scaler_X']
        self.scaler_y = checkpoint['scaler_y']
        
        self.mean_vgs = self.scaler_X.mean_[0]
        self.scale_vgs = self.scaler_X.scale_[0]
        self.mean_vds = self.scaler_X.mean_[1]
        self.scale_vds = self.scaler_X.scale_[1]
        self.mean_logids = self.scaler_y.mean_[0]
        self.scale_logids = self.scaler_y.scale_[0]
        
        if self.model_type == 'PhysicsAwareANN':
            self.w1 = checkpoint['w1']
            self.b1 = checkpoint['b1']
            self.w2 = checkpoint['w2']
            self.b2 = checkpoint['b2']
            self.w3 = checkpoint['w3']
            self.b3 = checkpoint['b3']
            self.vgs_scale = float(checkpoint['vgs_scale'])
            self.vds_sat = float(checkpoint['vds_sat'])
            self.hidden_size = checkpoint['hidden_size']
            self.n_layers = 3
        else:
            self.w1 = checkpoint['w1']
            self.b1 = checkpoint['b1']
            self.w2 = checkpoint['w2']
            self.b2 = checkpoint['b2']
            self.hidden_size = checkpoint['hidden_size']
            self.n_layers = 2
        
        print(f"加载模型成功: {self.model_type}")
        print(f"  - 隐藏层: {self.hidden_size} 个神经元")
        print(f"  - 网络层数: {self.n_layers}")
    
    def _generate_feature_code(self):
        """生成特征提取代码"""
        return f'''
        // 物理启发的特征提取
        // Vgs 特征: 线性, 平方, 指数
        real vgs_lin = Vgs_n;
        real vgs_sq = Vgs_n * Vgs_n;
        real vgs_exp = exp({self.vgs_scale:.6f} * Vgs_n) - 1.0;
        
        // Vds 特征: 线性, 平方, 饱和
        real vds_lin = Vds_n;
        real vds_sq = Vds_n * Vds_n;
        real vds_sat = tanh({self.vds_sat:.6f} * Vds_n);
'''
    
    def _generate_3layer_hidden(self, n_hidden):
        """生成三隐藏层网络代码"""
        # 生成第一层权重初始化
        w1_init = []
        for i in range(n_hidden):
            for j in range(6):
                w1_init.append(f'        w1[{i}][{j}] = {self.w1[i, j]:.8f};')
        
        b1_init = [f'        b1[{i}] = {self.b1[i]:.8f};' for i in range(n_hidden)]
        
        # 第二层权重
        w2_init = []
        for i in range(n_hidden):
            for j in range(n_hidden):
                w2_init.append(f'        w2[{i}][{j}] = {self.w2[i, j]:.8f};')
        
        b2_init = [f'        b2[{i}] = {self.b2[i]:.8f};' for i in range(n_hidden)]
        
        # 第三层权重
        w3_init = [f'        w3[0][{j}] = {self.w3[0, j]:.8f};' for j in range(n_hidden)]
        b3_init = f'        b3 = {self.b3[0]:.8f};'
        
        hidden_code = f'''
        // Hidden layer 1
        for (i = 0; i < {n_hidden}; i = i + 1) begin
            h1[i] = b1[i];
            for (j = 0; j < 6; j = j + 1)
                h1[i] = h1[i] + w1[i][j] * x[j];
            h1[i] = tanh(h1[i]);
        end
        
        // Hidden layer 2
        for (i = 0; i < {n_hidden}; i = i + 1) begin
            h2[i] = b2[i];
            for (j = 0; j < {n_hidden}; j = j + 1)
                h2[i] = h2[i] + w2[i][j] * h1[j];
            h2[i] = tanh(h2[i]);
        end
        
        // Output layer
        out = b3;
        for (j = 0; j < {n_hidden}; j = j + 1)
            out = out + w3[0][j] * h2[j];
'''
        
        # 初始化代码
        init_code = '\n'.join(w1_init) + '\n\n' + '\n'.join(b1_init) + '\n\n'
        init_code += '\n'.join(w2_init) + '\n\n' + '\n'.join(b2_init) + '\n\n'
        init_code += '\n'.join(w3_init) + '\n\n' + b3_init
        
        return hidden_code, init_code
    
    def _generate_2layer_hidden(self, n_hidden):
        """生成两隐藏层网络代码"""
        # 第一层权重初始化
        w1_init = []
        for i in range(n_hidden):
            for j in range(5):  # SimplePowerLawANN 有5个输入特征
                w1_init.append(f'        w1[{i}][{j}] = {self.w1[i, j]:.8f};')
        
        b1_init = [f'        b1[{i}] = {self.b1[i]:.8f};' for i in range(n_hidden)]
        
        # 输出层权重
        w2_init = [f'        w2[0][{j}] = {self.w2[0, j]:.8f};' for j in range(n_hidden)]
        b2_init = f'        b2 = {self.b2[0]:.8f};'
        
        hidden_code = f'''
        // Hidden layer
        for (i = 0; i < {n_hidden}; i = i + 1) begin
            h1[i] = b1[i];
            for (j = 0; j < 5; j = j + 1)
                h1[i] = h1[i] + w1[i][j] * x[j];
            h1[i] = tanh(h1[i]);
        end
        
        // Output layer
        out = b2;
        for (j = 0; j < {n_hidden}; j = j + 1)
            out = out + w2[0][j] * h1[j];
'''
        
        # 初始化代码
        init_code = '\n'.join(w1_init) + '\n\n' + '\n'.join(b1_init) + '\n\n'
        init_code += '\n'.join(w2_init) + '\n\n' + b2_init
        
        return hidden_code, init_code
    
    def generate_verilog_a(self, output_file='ann_improved.va'):
        """生成 Verilog-A 模型"""
        
        n_hidden = self.hidden_size
        
        # 生成特征提取代码
        feature_code = self._generate_feature_code()
        
        # 生成隐藏层代码
        if self.n_layers == 3:
            hidden_code, init_code = self._generate_3layer_hidden(n_hidden)
            hidden_vars = f'    real h1[{n_hidden}];\n    real h2[{n_hidden}];'
        else:
            hidden_code, init_code = self._generate_2layer_hidden(n_hidden)
            hidden_vars = f'    real h1[{n_hidden}];'
        
        va_code = f'''// ANN-based MOSFET Model - Physics-Aware Design
// Vgs features: linear, square, exponential
// Vds features: linear, square, saturation
// Generated from PyTorch training

`include "disciplines.vams"

module ann_improved(d, g, s);
    inout d, g, s;
    electrical d, g, s;
    
    parameter real w = 10e-6;
    parameter real l = 10e-6;
    
    real Vgs, Vds, Ids;
    real logIds;
    real Vgs_n, Vds_n;
    real out;
    
    // Normalization parameters
    real mean_vgs = {self.mean_vgs:.6f};
    real scale_vgs = {self.scale_vgs:.6f};
    real mean_vds = {self.mean_vds:.6f};
    real scale_vds = {self.scale_vds:.6f};
    real mean_logids = {self.mean_logids:.6f};
    real scale_logids = {self.scale_logids:.6f};
    
    // Neural network weights
    real w1[{n_hidden}][6];
    real b1[{n_hidden}];
'''
        
        if self.n_layers == 3:
            va_code += f'''    real w2[{n_hidden}][{n_hidden}];
    real b2[{n_hidden}];
    real w3[1][{n_hidden}];
    real b3;
'''
        else:
            va_code += f'''    real w2[1][{n_hidden}];
    real b2;
'''
        
        va_code += f'''
{hidden_vars}
    integer i, j;
    
    // Initialize weights (in analog initial block)
    analog initial begin
{init_code}
    end
    
    analog begin
        // Get input voltages
        Vgs = V(g, s);
        Vds = V(d, s);
        
        // Normalize inputs
        Vgs_n = (Vgs - mean_vgs) / scale_vgs;
        Vds_n = (Vds - mean_vds) / scale_vds;
        
        {feature_code}
        
        // Input features
        real x[6];
        x[0] = vgs_lin;
        x[1] = vgs_sq;
        x[2] = vgs_exp;
        x[3] = vds_lin;
        x[4] = vds_sq;
        x[5] = vds_sat;
        
{hidden_code}
        
        // Denormalize
        logIds = out * scale_logids + mean_logids;
        Ids = pow(10, logIds) * (w / l);
        
        // Output current
        I(d, s) <+ Ids;
        I(g, s) <+ 0;
    end
endmodule
'''
        
        with open(output_file, 'w') as f:
            f.write(va_code)
        
        print(f"\nVerilog-A模型已保存到 {output_file}")
        print(f"  - 特征: Vgs(lin,sq,exp) + Vds(lin,sq,sat)")
        print(f"  - 隐藏层: {n_hidden} 个神经元")
        print(f"  - 网络层数: {self.n_layers}")
        
        return va_code


def main():
    print("="*60)
    print("导出物理启发的 ANN Verilog-A 模型")
    print("="*60)
    
    try:
        exporter = ImprovedVerilogAExporter('ann_improved_model.pkl')
        exporter.generate_verilog_a('ann_improved.va')
        
        print("\n" + "="*60)
        print("下一步操作:")
        print("="*60)
        print("\n1. 编译: openvaf ann_improved.va")
        print("\n2. 测试网表 test_improved.cir:")
        print('''
* Test Improved ANN Model

.control
pre_osdi ann_improved.osdi
.endc

.model NMOS1 ann_improved(w=10e-6 l=10e-6)

N1 d_int g 0 NMOS1
Vmeas d d_int 0

Vds d 0 2
Vgs g 0 0

.dc Vgs 0 3 0.05

.options savecurrents

.control
run
plot i(Vmeas)
write improved_test.txt v(g) i(Vmeas)
.endc

.end
''')
        print("3. 仿真: ngspice test_improved.cir")
        
    except FileNotFoundError:
        print("错误: 未找到 ann_improved_model.pkl")
        print("请先运行 train_ann_weighted.py 训练模型")
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()