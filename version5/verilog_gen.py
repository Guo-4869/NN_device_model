# fixed_verilog_gen.py
"""
修复版Verilog-A生成器 - 32x32网络
生成优化的Verilog-A代码
"""

import torch
import numpy as np
from fix_nn_model import FixedBSIM_NN


class FixedVerilogAGenerator:
    """优化版Verilog-A生成器"""
    
    def __init__(self, model, input_means, input_stds):
        self.model = model
        
        # 归一化参数
        self.input_means = input_means
        self.input_stds = input_stds
        
        # 提取权重（使用eval模式）
        self.model.eval()
        
        # 获取网络层
        self.w1 = model.fc1.weight.detach().cpu().numpy()  # [32, 3]
        self.b1 = model.fc1.bias.detach().cpu().numpy()    # [32]
        self.w2 = model.fc2.weight.detach().cpu().numpy()  # [32, 32]
        self.b2 = model.fc2.bias.detach().cpu().numpy()    # [32]
        self.w3 = model.fc3.weight.detach().cpu().numpy()  # [1, 32]
        self.b3 = model.fc3.bias.detach().cpu().numpy()    # [1]
        
        # 数值精度
        self.decimals = 8
        
        # 阈值：跳过绝对值小于此值的权重
        self.weight_threshold = 1e-6
        
    def _format_val(self, val):
        """格式化数值"""
        if abs(val) < self.weight_threshold:
            return None
        return f"{val:.{self.decimals}f}"
    
    def _isru(self, x):
        """ISRU内联表达式"""
        return f"({x})/sqrt(1+({x})*({x}))"
    
    def generate(self, output_file='fixed_bsim_nn.va'):
        """生成Verilog-A代码"""
        
        lines = [
            '// Fixed BSIM-NN Compact Model (32x32 network)',
            '// Network: 3 inputs -> 32 neurons -> 32 neurons -> 1 output',
            '// Activation: ISRU(x) = x / sqrt(1 + x^2)',
            '//',
            '`include "disciplines.vams"',
            '`include "constants.vams"',
            '',
            'module bsim_nn(d, g, s, b);',
            '    inout d, g, s, b;',
            '    electrical d, g, s, b;',
            '',
            '    parameter real L = 1e-6 from (0:inf);',
            '    parameter real W = 10e-6 from (0:inf);',
            '',
            '    // Internal variables',
            '    real VGS, VDS, VBS, VGS_n, VDS_n, VBS_n, Ids, log10_Id;',
        ]
        
        # 隐藏层变量
        h1_vars = [f'h1_{i}' for i in range(32)]
        h2_vars = [f'h2_{i}' for i in range(32)]
        lines.append(f'    real {", ".join(h1_vars)};')
        lines.append(f'    real {", ".join(h2_vars)};')
        
        # 归一化参数
        lines.extend([
            '',
            f'    real mean_vgs = {self._format_val(self.input_means[0])};',
            f'    real std_vgs  = {self._format_val(self.input_stds[0])};',
            f'    real mean_vds = {self._format_val(self.input_means[1])};',
            f'    real std_vds  = {self._format_val(self.input_stds[1])};',
            f'    real mean_vbs = {self._format_val(self.input_means[2])};',
            f'    real std_vbs  = {self._format_val(self.input_stds[2])};',
            '',
            '    analog begin',
            '        // Read terminal voltages',
            '        VGS = V(g, s);',
            '        VDS = V(d, s);',
            '        VBS = V(b, s);',
            '',
            '        // Normalize inputs',
            '        VGS_n = (VGS - mean_vgs) / std_vgs;',
            '        VDS_n = (VDS - mean_vds) / std_vds;',
            '        VBS_n = (VBS - mean_vbs) / std_vbs;',
            '',
            '        // Hidden Layer 1 (32 neurons with ISRU)',
        ])
        
        # 第一隐藏层
        for i in range(32):
            # 构建线性组合
            terms = []
            b_val = self._format_val(self.b1[i])
            if b_val:
                terms.append(b_val)
            
            w0 = self._format_val(self.w1[i, 0])
            if w0:
                terms.append(f'{w0}*VGS_n')
            
            w1 = self._format_val(self.w1[i, 1])
            if w1:
                terms.append(f'{w1}*VDS_n')
            
            w2 = self._format_val(self.w1[i, 2])
            if w2:
                terms.append(f'{w2}*VBS_n')
            
            expr = '+'.join(terms)
            lines.append(f'        h1_{i} = {self._isru(expr)};')
        
        lines.append('')
        lines.append('        // Hidden Layer 2 (32 neurons with ISRU)')
        
        # 第二隐藏层
        for i in range(32):
            terms = [self._format_val(self.b2[i])]
            for j in range(32):
                w = self._format_val(self.w2[i, j])
                if w:
                    terms.append(f'{w}*h1_{j}')
            expr = '+'.join([t for t in terms if t])
            lines.append(f'        h2_{i} = {self._isru(expr)};')
        
        lines.append('')
        lines.append('        // Output Layer (linear)')
        
        # 输出层
        terms = [self._format_val(self.b3[0])]
        for j in range(32):
            w = self._format_val(self.w3[0, j])
            if w:
                terms.append(f'{w}*h2_{j}')
        expr = '+'.join([t for t in terms if t])
        lines.append(f'        log10_Id = {expr};')
        
        lines.extend([
            '',
            '        // Clamp and convert to current',
            '        log10_Id = (log10_Id < -12) ? -12 : ((log10_Id > -3) ? -3 : log10_Id);',
            '        Ids = pow(10, log10_Id) * (W / 10e-6);',
            '',
            '        // Physical constraints',
            '        if (Ids < 1e-12) Ids = 1e-12;',
            '        if (Ids > 1e-3) Ids = 1e-3;',
            '        if (VDS == 0) Ids = 0;',
            '',
            '        // Output currents',
            '        I(d, s) <+ Ids;',
            '        I(g, s) <+ 0;',
            '        I(b, s) <+ 0;',
            '    end',
            '',
            'endmodule'
        ])
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"Verilog-A model generated: {output_file}")
        
        # 统计
        with open(output_file, 'r') as f:
            lines_count = len(f.readlines())
        print(f"Total lines: {lines_count}")
        
        return output_file


def export_weights_to_json(model, output_file='fixed_weights.json'):
    """导出权重到JSON"""
    import json
    
    model.eval()
    
    weights_dict = {
        'input_means': model.input_norm.running_mean.numpy().tolist(),
        'input_stds': np.sqrt(model.input_norm.running_var.numpy()).tolist(),
        'layer_sizes': [3, 32, 32, 1],
        'activation': 'isru'
    }
    
    # 添加权重
    weights_dict['fc1_weight'] = model.fc1.weight.detach().cpu().numpy().tolist()
    weights_dict['fc1_bias'] = model.fc1.bias.detach().cpu().numpy().tolist()
    weights_dict['fc2_weight'] = model.fc2.weight.detach().cpu().numpy().tolist()
    weights_dict['fc2_bias'] = model.fc2.bias.detach().cpu().numpy().tolist()
    weights_dict['fc3_weight'] = model.fc3.weight.detach().cpu().numpy().tolist()
    weights_dict['fc3_bias'] = model.fc3.bias.detach().cpu().numpy().tolist()
    
    with open(output_file, 'w') as f:
        json.dump(weights_dict, f, indent=2)
    
    print(f"Weights exported to {output_file}")


def generate_model(checkpoint_path='fixed_best_model.pth', output_va='fixed_bsim_nn.va'):
    """主函数"""
    
    print("="*60)
    print("Fixed Verilog-A Model Generator (32x32)")
    print("="*60)
    
    # 加载模型
    model = FixedBSIM_NN(input_dim=3, hidden_size=32, num_layers=2)
    
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        print(f"✓ Loaded model from {checkpoint_path}")
    except FileNotFoundError:
        print(f"✗ Error: {checkpoint_path} not found!")
        print("\nPlease run fixed_nn_model.py first to train the model.")
        return None
    
    model.eval()
    
    # 获取归一化参数
    if hasattr(model, 'input_norm'):
        input_means = model.input_norm.running_mean.numpy()
        input_stds = np.sqrt(model.input_norm.running_var.numpy())
        print(f"\nNormalization parameters:")
        print(f"  VGS: mean={input_means[0]:.4f}, std={input_stds[0]:.4f}")
        print(f"  VDS: mean={input_means[1]:.4f}, std={input_stds[1]:.4f}")
        print(f"  VBS: mean={input_means[2]:.4f}, std={input_stds[2]:.4f}")
    else:
        input_means = np.array([0.5, 0.5, 0.0])
        input_stds = np.array([0.3, 0.3, 0.1])
        print("Using default normalization parameters")
    
    # 生成Verilog-A
    generator = FixedVerilogAGenerator(model, input_means, input_stds)
    generator.generate(output_va)
    
    # 导出JSON备份
    export_weights_to_json(model, 'fixed_weights.json')
    
    print("\n" + "="*60)
    print("Compilation Instructions:")
    print("="*60)
    print(f"  openvaf {output_va} --ngspice -o bsim_nn.so")
    print("\nIf compilation is too slow, try:")
    print("  openvaf --opt-speed fixed_bsim_nn.va --ngspice -o bsim_nn.so")
    print("="*60)
    
    return output_va


if __name__ == "__main__":
    generate_model('fixed_best_model.pth', 'fixed_bsim_nn.va')