import numpy as np
import torch


class VerilogAExporter:
    """
    生成符合 OpenVAF 语法的 Verilog-A 模型
    使用 0-based 数组索引 [0:size-1]
    不使用 ` 宏，使用标准函数
    """
    
    def __init__(self, model_path='ann_bsim_model.pth'):
        checkpoint = torch.load(model_path, map_location='cpu')
        self.scaler_X = checkpoint['scaler_X']
        self.scaler_y = checkpoint['scaler_y']
        self.weights = checkpoint['weights']
        self.biases = checkpoint['biases']
        self.hidden_sizes = checkpoint['hidden_sizes']
        
    def _format_array_1d(self, arr, precision=8):
        """格式化1D数组为Verilog-A格式"""
        return ', '.join([f"{x:.{precision}e}" for x in arr.flatten()])
    
    def _format_array_2d(self, arr, precision=8):
        """格式化2D数组为Verilog-A格式 - 使用0-based索引"""
        rows, cols = arr.shape
        lines = []
        for i in range(rows):
            row_str = ', '.join([f"{x:.{precision}e}" for x in arr[i, :]])
            lines.append(f"        w1[{i}][0:{cols-1}] = '{{{row_str}}};")
        return '\n'.join(lines)
    
    def _format_weights_2d(self, arr, name, precision=8):
        """格式化权重矩阵 - 每个元素单独赋值（更兼容）"""
        rows, cols = arr.shape
        lines = []
        for i in range(rows):
            for j in range(cols):
                lines.append(f"        {name}[{i}][{j}] = {arr[i, j]:.{precision}e};")
        return '\n'.join(lines)
    
    def generate_verilog_a(self, output_file='ann_bsim.va'):
        """生成符合OpenVAF语法的Verilog-A模型"""
        
        # 获取归一化参数
        mean_x = self.scaler_X.mean_
        scale_x = self.scaler_X.scale_
        mean_y = self.scaler_y.mean_[0]
        scale_y = self.scaler_y.scale_[0]
        
        # 网络结构
        n_input = 5  # Vgs, Vds, log10(W), log10(L), Temp
        n_hidden1 = self.weights[0].shape[0]  # 32
        n_hidden2 = self.weights[1].shape[0]  # 32
        n_hidden3 = self.weights[2].shape[0]  # 32
        n_output = self.weights[3].shape[0]   # 1
        
        # 提取权重和偏置
        w1 = self.weights[0]  # [32, 5]
        b1 = self.biases[0]    # [32]
        w2 = self.weights[1]  # [32, 32]
        b2 = self.biases[1]    # [32]
        w3 = self.weights[2]  # [32, 32]
        b3 = self.biases[2]    # [32]
        w4 = self.weights[3]  # [1, 32]
        b4 = self.biases[3]    # [1]
        
        # 生成Verilog-A代码
        va_code = f'''// ANN-based MOSFET Model trained on BSIM4 data
// Generated for OpenVAF compiler
// Network: {n_input} -> {n_hidden1} -> {n_hidden2} -> {n_hidden3} -> {n_output}

`include "disciplines.vams"

module ann_bsim(d, g, s, b);
    inout d, g, s, b;
    electrical d, g, s, b;
    
    // Geometry parameters
    parameter real w = 10e-6;
    parameter real l = 1e-6;
    
    // Normalization parameters
    real mean_vgs = {mean_x[0]:.10e};
    real mean_vds = {mean_x[1]:.10e};
    real mean_logw = {mean_x[2]:.10e};
    real mean_logl = {mean_x[3]:.10e};
    real mean_temp = {mean_x[4]:.10e};
    
    real scale_vgs = {scale_x[0]:.10e};
    real scale_vds = {scale_x[1]:.10e};
    real scale_logw = {scale_x[2]:.10e};
    real scale_logl = {scale_x[3]:.10e};
    real scale_temp = {scale_x[4]:.10e};
    
    real mean_logids = {mean_y:.10e};
    real scale_logids = {scale_y:.10e};
    
    // Neural network weights and biases
    real w1[{n_hidden1}][{n_input}];
    real b1[{n_hidden1}];
    real w2[{n_hidden2}][{n_hidden1}];
    real b2[{n_hidden2}];
    real w3[{n_hidden3}][{n_hidden2}];
    real b3[{n_hidden3}];
    real w4[{n_output}][{n_hidden3}];
    real b4[{n_output}];
    
    // Local variables
    real logw, logl;
    real x[{n_input}];
    real h1[{n_hidden1}];
    real h2[{n_hidden2}];
    real h3[{n_hidden3}];
    real out;
    integer i, j;
    
    // Model signals
    real vgs, vds, logids, ids;
    
    // Initialize weights and biases
    analog initial begin
        // Layer 1 weights (input -> hidden1)
'''
        
        # 添加 Layer 1 权重
        for i in range(w1.shape[0]):
            for j in range(w1.shape[1]):
                va_code += f'        w1[{i}][{j}] = {w1[i, j]:.10e};\n'
        
        va_code += '\n        // Layer 1 biases\n'
        for i in range(len(b1)):
            va_code += f'        b1[{i}] = {b1[i]:.10e};\n'
        
        # Layer 2 权重
        va_code += '\n        // Layer 2 weights (hidden1 -> hidden2)\n'
        for i in range(w2.shape[0]):
            for j in range(w2.shape[1]):
                va_code += f'        w2[{i}][{j}] = {w2[i, j]:.10e};\n'
        
        va_code += '\n        // Layer 2 biases\n'
        for i in range(len(b2)):
            va_code += f'        b2[{i}] = {b2[i]:.10e};\n'
        
        # Layer 3 权重
        va_code += '\n        // Layer 3 weights (hidden2 -> hidden3)\n'
        for i in range(w3.shape[0]):
            for j in range(w3.shape[1]):
                va_code += f'        w3[{i}][{j}] = {w3[i, j]:.10e};\n'
        
        va_code += '\n        // Layer 3 biases\n'
        for i in range(len(b3)):
            va_code += f'        b3[{i}] = {b3[i]:.10e};\n'
        
        # Layer 4 权重
        va_code += '\n        // Layer 4 weights (hidden3 -> output)\n'
        for i in range(w4.shape[0]):
            for j in range(w4.shape[1]):
                va_code += f'        w4[{i}][{j}] = {w4[i, j]:.10e};\n'
        
        va_code += '\n        // Layer 4 biases\n'
        for i in range(len(b4)):
            va_code += f'        b4[{i}] = {b4[i]:.10e};\n'
        
        # Forward propagation 函数
        va_code += f'''
    end
    
    // Forward propagation function
    real function forward(real vgs_val, real vds_val, real w_val, real l_val);
    begin
        // Log transform of geometry using natural log
        logw = ln(w_val) / 2.302585092994046;
        logl = ln(l_val) / 2.302585092994046;
        
        // Normalize inputs
        x[0] = (vgs_val - mean_vgs) / scale_vgs;
        x[1] = (vds_val - mean_vds) / scale_vds;
        x[2] = (logw - mean_logw) / scale_logw;
        x[3] = (logl - mean_logl) / scale_logl;
        x[4] = 0.0;  // Temperature (fixed at 27C)
        
        // Hidden layer 1
        for (i = 0; i < {n_hidden1}; i = i + 1) begin
            h1[i] = b1[i];
            for (j = 0; j < {n_input}; j = j + 1)
                h1[i] = h1[i] + w1[i][j] * x[j];
            // Clamp to avoid overflow
            if (h1[i] > 10) h1[i] = 10;
            if (h1[i] < -10) h1[i] = -10;
            h1[i] = tanh(h1[i]);
        end
        
        // Hidden layer 2
        for (i = 0; i < {n_hidden2}; i = i + 1) begin
            h2[i] = b2[i];
            for (j = 0; j < {n_hidden1}; j = j + 1)
                h2[i] = h2[i] + w2[i][j] * h1[j];
            if (h2[i] > 10) h2[i] = 10;
            if (h2[i] < -10) h2[i] = -10;
            h2[i] = tanh(h2[i]);
        end
        
        // Hidden layer 3
        for (i = 0; i < {n_hidden3}; i = i + 1) begin
            h3[i] = b3[i];
            for (j = 0; j < {n_hidden2}; j = j + 1)
                h3[i] = h3[i] + w3[i][j] * h2[j];
            if (h3[i] > 10) h3[i] = 10;
            if (h3[i] < -10) h3[i] = -10;
            h3[i] = tanh(h3[i]);
        end
        
        // Output layer
        out = b4[0];
        for (j = 0; j < {n_hidden3}; j = j + 1)
            out = out + w4[0][j] * h3[j];
        
        // Denormalize output
        forward = out * scale_logids + mean_logids;
    end
    endfunction
    
    analog begin
        vgs = V(g, s);
        vds = V(d, s);
        
        // Calculate log10(Ids) using neural network
        logids = forward(vgs, vds, w, l);
        
        // Convert to linear current
        ids = pow(10, logids);
        
        // Limit to realistic range
        if (ids < 1e-15) ids = 1e-15;
        if (ids > 0.1) ids = 0.1;
        
        I(d, s) <+ ids;
        I(g, s) <+ 0;
        I(b, s) <+ 0;
    end
endmodule
'''
        
        with open(output_file, 'w') as f:
            f.write(va_code)
        
        print(f"Verilog-A模型已保存到 {output_file}")
        print(f"网络结构: {n_input}输入 -> {n_hidden1} -> {n_hidden2} -> {n_hidden3} -> {n_output}输出")
        print(f"总参数量: {w1.size + b1.size + w2.size + b2.size + w3.size + b3.size + w4.size + b4.size}")
        
        return va_code


def create_test_netlist():
    """创建测试网表"""
    test_cir = '''* Test circuit for ANN BSIM model

.control
pre_osdi ann_bsim.osdi
.endc

.model ANN_NMOS ann_bsim(w=10e-6 l=1e-6)

N1 d g 0 0 ANN_NMOS

Vds d 0 0
Vgs g 0 0

.dc Vgs 0 3 0.05 Vds 0 3 0.5

.options savecurrents

.control
run
plot vds#branch
write ann_test.txt v(d) v(g) vds#branch
.endc

.end
'''
    with open('test_ann_bsim.cir', 'w') as f:
        f.write(test_cir)
    print("测试网表已保存到 test_ann_bsim.cir")


if __name__ == "__main__":
    try:
        exporter = VerilogAExporter('ann_bsim_model.pth')
        exporter.generate_verilog_a('ann_bsim.va')
        create_test_netlist()
        print("\n" + "="*50)
        print("下一步操作:")
        print("1. 编译: openvaf ann_bsim.va")
        print("2. 仿真: ngspice test_ann_bsim.cir")
        print("="*50)
    except FileNotFoundError:
        print("错误: 未找到 ann_bsim_model.pth")
        print("请先运行 train_ann_bsim.py 训练模型")
        
        # 创建简化版模型用于测试
        print("\n创建简化版模型用于测试...")
        simple_va = '''`include "disciplines.vams"

module ann_bsim(d, g, s, b);
    inout d, g, s, b;
    electrical d, g, s, b;
    
    parameter real w = 10e-6;
    parameter real l = 1e-6;
    parameter real beta = 1e-3;
    parameter real vth = 0.7;
    
    real vgs, vds, ids;
    
    analog begin
        vgs = V(g, s);
        vds = V(d, s);
        
        if (vgs <= vth) begin
            ids = 0;
        end else if (vds < (vgs - vth)) begin
            ids = beta * (w/l) * (2*(vgs - vth)*vds - vds*vds);
        end else begin
            ids = beta * (w/l) * (vgs - vth)*(vgs - vth);
        end
        
        I(d, s) <+ ids;
        I(g, s) <+ 0;
        I(b, s) <+ 0;
    end
endmodule
'''
        with open('ann_bsim.va', 'w') as f:
            f.write(simple_va)
        print("已创建简化版 ann_bsim.va (使用平方律模型)")