# generate_verilog.py
"""
将训练好的BSIM-NN PyTorch模型转换为Verilog-A代码
基于论文: "A SPICE-compatible Neural Network Compact Model for Efficient IC Simulations"

网络结构: 3输入 -> 32神经元 -> 32神经元 -> 1输出
激活函数: ISRU (x / sqrt(1 + x^2))
输出: log10(ID)
"""

import torch
import numpy as np
import json
from train_nn_model import BSIM_NN_IV


class VerilogAGenerator:
    """Verilog-A代码生成器"""
    
    def __init__(self, model, input_means, input_stds):
        """
        参数:
            model: 训练好的PyTorch模型
            input_means: 输入均值 [VGS_mean, VDS_mean, VBS_mean]
            input_stds: 输入标准差 [VGS_std, VDS_std, VBS_std]
        """
        self.model = model
        self.input_means = input_means
        self.input_stds = input_stds
        
        # 提取网络权重
        self.weights = []
        self.biases = []
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                self.weights.append(param.detach().cpu().numpy())
            elif 'bias' in name:
                self.biases.append(param.detach().cpu().numpy())
        
        # 网络结构
        self.layer_sizes = [3, 32, 32, 1]  # 输入, 隐藏1, 隐藏2, 输出
        
    def _isru_inline(self, x_expr):
        """生成ISRU内联表达式: x / sqrt(1 + x^2)"""
        return f"({x_expr} / `sqrt(1.0 + ({x_expr})*({x_expr})))"
    
    def generate(self, output_file='bsim_nn.va'):
        """生成Verilog-A代码"""
        
        code_lines = []
        
        # ========== 文件头 ==========
        code_lines.append("// ============================================================")
        code_lines.append("// BSIM-NN: Neural Network Based MOSFET Compact Model")
        code_lines.append("// ============================================================")
        code_lines.append("// Reference: C.T. Tung et al., IEEE TED, 2023")
        code_lines.append("//           \"A SPICE-compatible Neural Network Compact Model\"")
        code_lines.append("// ============================================================")
        code_lines.append("// Network Architecture:")
        code_lines.append(f"//   Input Layer: {self.layer_sizes[0]} nodes")
        code_lines.append(f"//   Hidden Layer 1: {self.layer_sizes[1]} nodes (ISRU)")
        code_lines.append(f"//   Hidden Layer 2: {self.layer_sizes[2]} nodes (ISRU)")
        code_lines.append(f"//   Output Layer: {self.layer_sizes[3]} node (linear)")
        code_lines.append("// ============================================================")
        code_lines.append("// Activation: ISRU(x) = x / sqrt(1 + x^2) (inline expansion)")
        code_lines.append("// Output: log10(ID) = f(VGS, VDS, VBS)")
        code_lines.append("// ============================================================")
        code_lines.append("")
        code_lines.append("`include \"disciplines.vams\"")
        code_lines.append("`include \"constants.vams\"")
        code_lines.append("")
        code_lines.append("module bsim_nn(d, g, s, b);")
        code_lines.append("    inout d, g, s, b;")
        code_lines.append("    electrical d, g, s, b;")
        code_lines.append("")
        
        # ========== 参数定义 ==========
        code_lines.append("    // ==============================================")
        code_lines.append("    // Geometry Parameters")
        code_lines.append("    // ==============================================")
        code_lines.append("    parameter real L = 1e-6 from (0:inf);   // Channel length (m)")
        code_lines.append("    parameter real W = 10e-6 from (0:inf);  // Channel width (m)")
        code_lines.append("    parameter real NFIN = 1;                // Number of fins")
        code_lines.append("")
        code_lines.append("    // ==============================================")
        code_lines.append("    // Internal Variables")
        code_lines.append("    // ==============================================")
        code_lines.append("    real VGS, VDS, VBS;")
        code_lines.append("    real Ids, Igs, Ibs;")
        code_lines.append("    real log10_Id;")
        code_lines.append("    real gm, gds;")
        code_lines.append("")
        
        # ========== 输入归一化 ==========
        code_lines.append("    // ==============================================")
        code_lines.append("    // Input Normalization (Z-score)")
        code_lines.append("    // ==============================================")
        code_lines.append(f"    real mean_vgs = {self.input_means[0]:.8f};")
        code_lines.append(f"    real std_vgs  = {self.input_stds[0]:.8f};")
        code_lines.append(f"    real mean_vds = {self.input_means[1]:.8f};")
        code_lines.append(f"    real std_vds  = {self.input_stds[1]:.8f};")
        code_lines.append(f"    real mean_vbs = {self.input_means[2]:.8f};")
        code_lines.append(f"    real std_vbs  = {self.input_stds[2]:.8f};")
        code_lines.append("")
        code_lines.append("    real VGS_n, VDS_n, VBS_n;")
        code_lines.append("")
        
        # ========== 隐藏层节点定义 ==========
        h1_size = self.layer_sizes[1]
        h2_size = self.layer_sizes[2]
        
        h1_vars = [f"h1_{i}" for i in range(h1_size)]
        h2_vars = [f"h2_{i}" for i in range(h2_size)]
        
        code_lines.append("    // ==============================================")
        code_lines.append(f"    // Hidden Layer 1: {h1_size} neurons (ISRU)")
        code_lines.append("    // ==============================================")
        code_lines.append(f"    real {', '.join(h1_vars)};")
        code_lines.append("")
        code_lines.append("    // ==============================================")
        code_lines.append(f"    // Hidden Layer 2: {h2_size} neurons (ISRU)")
        code_lines.append("    // ==============================================")
        code_lines.append(f"    real {', '.join(h2_vars)};")
        code_lines.append("")
        code_lines.append("    // ==============================================")
        code_lines.append("    // Output Layer")
        code_lines.append("    // ==============================================")
        code_lines.append("    real log10_Id_nn;")
        code_lines.append("")
        
        # ========== 前向传播 ==========
        code_lines.append("    // ==============================================")
        code_lines.append("    // Forward Propagation")
        code_lines.append("    // ==============================================")
        code_lines.append("")
        
        # 输入归一化
        code_lines.append("    // Step 1: Normalize inputs")
        code_lines.append("    VGS_n = (VGS - mean_vgs) / std_vgs;")
        code_lines.append("    VDS_n = (VDS - mean_vds) / std_vds;")
        code_lines.append("    VBS_n = (VBS - mean_vbs) / std_vbs;")
        code_lines.append("")
        
        # 第一隐藏层 - 使用内联ISRU
        code_lines.append("    // Step 2: Hidden Layer 1 (ISRU activation inline)")
        w1 = self.weights[0]  # shape: [32, 3]
        b1 = self.biases[0]   # shape: [32]
        
        for i in range(h1_size):
            # 计算线性组合
            linear_expr = f"{b1[i]:.10f}"
            linear_expr += f" + {w1[i, 0]:.10f}*VGS_n"
            linear_expr += f" + {w1[i, 1]:.10f}*VDS_n"
            linear_expr += f" + {w1[i, 2]:.10f}*VBS_n"
            # 应用内联ISRU
            code_lines.append(f"        h1_{i} = {self._isru_inline(linear_expr)};")
        code_lines.append("")
        
        # 第二隐藏层 - 使用内联ISRU
        code_lines.append("    // Step 3: Hidden Layer 2 (ISRU activation inline)")
        w2 = self.weights[1]  # shape: [32, 32]
        b2 = self.biases[1]   # shape: [32]
        
        for i in range(h2_size):
            linear_expr = f"{b2[i]:.10f}"
            for j in range(h1_size):
                linear_expr += f" + {w2[i, j]:.10f}*h1_{j}"
            code_lines.append(f"        h2_{i} = {self._isru_inline(linear_expr)};")
        code_lines.append("")
        
        # 输出层 (线性，无激活)
        code_lines.append("    // Step 4: Output Layer (linear)")
        w3 = self.weights[2]  # shape: [1, 32]
        b3 = self.biases[2]   # shape: [1]
        
        linear_expr = f"{b3[0]:.10f}"
        for j in range(h2_size):
            linear_expr += f" + {w3[0, j]:.10f}*h2_{j}"
        code_lines.append(f"        log10_Id_nn = {linear_expr};")
        code_lines.append("")
        
        # ========== 输出计算 ==========
        code_lines.append("    // ==============================================")
        code_lines.append("    // Output Calculation")
        code_lines.append("    // ==============================================")
        code_lines.append("    // Clamp output to reasonable range")
        code_lines.append("    log10_Id = (log10_Id_nn < -12) ? -12 : ((log10_Id_nn > -3) ? -3 : log10_Id_nn);")
        code_lines.append("    ")
        code_lines.append("    // Convert from log10 to linear current")
        code_lines.append("    Ids = pow(10, log10_Id);")
        code_lines.append("    ")
        code_lines.append("    // Apply geometry scaling")
        code_lines.append("    Ids = Ids * (W / 10e-6);")
        code_lines.append("")
        
        # ========== 物理约束 ==========
        code_lines.append("    // ==============================================")
        code_lines.append("    // Physical Constraints")
        code_lines.append("    // ==============================================")
        code_lines.append("    if (Ids < 1e-12) Ids = 1e-12;")
        code_lines.append("    if (Ids > 1e-3) Ids = 1e-3;")
        code_lines.append("    if (VDS == 0) Ids = 0;")
        code_lines.append("    ")
        code_lines.append("    // Gate and body currents (negligible for DC)")
        code_lines.append("    Igs = 0;")
        code_lines.append("    Ibs = 0;")
        code_lines.append("")
        
        # ========== 跨导计算 ==========
        code_lines.append("    // ==============================================")
        code_lines.append("    // Transconductance (for AC analysis)")
        code_lines.append("    // ==============================================")
        code_lines.append("    gm = ddx(Ids, VGS);")
        code_lines.append("    gds = ddx(Ids, VDS);")
        code_lines.append("")
        
        # ========== Analog Block ==========
        code_lines.append("    // ==============================================")
        code_lines.append("    // Analog Behavior Block")
        code_lines.append("    // ==============================================")
        code_lines.append("    analog begin")
        code_lines.append("        // Read terminal voltages")
        code_lines.append("        VGS = V(g, s);")
        code_lines.append("        VDS = V(d, s);")
        code_lines.append("        VBS = V(b, s);")
        code_lines.append("        ")
        code_lines.append("        // Calculate drain current (from NN above)")
        code_lines.append("        ")
        code_lines.append("        // Assign currents to terminals")
        code_lines.append("        I(d, s) <+ Ids;")
        code_lines.append("        I(g, s) <+ Igs;")
        code_lines.append("        I(b, s) <+ Ibs;")
        code_lines.append("    end")
        code_lines.append("")
        code_lines.append("endmodule")
        code_lines.append("")
        code_lines.append("// ============================================================")
        code_lines.append("// End of BSIM-NN Verilog-A Model")
        code_lines.append("// ============================================================")
        
        # 写入文件
        with open(output_file, 'w') as f:
            f.write('\n'.join(code_lines))
        
        print(f"Verilog-A model generated: {output_file}")
        return output_file


def export_weights_to_json(model, input_means, input_stds, output_file='nn_weights.json'):
    """导出权重为JSON格式（用于调试或备用）"""
    
    weights_dict = {
        'input_means': input_means.tolist(),
        'input_stds': input_stds.tolist(),
        'layer_sizes': [3, 32, 32, 1],
        'activation': 'isru'
    }
    
    # 提取所有权重
    for i, (name, param) in enumerate(model.named_parameters()):
        if 'weight' in name:
            weights_dict[f'weight_{i}'] = param.detach().cpu().numpy().tolist()
        elif 'bias' in name:
            weights_dict[f'bias_{i}'] = param.detach().cpu().numpy().tolist()
    
    with open(output_file, 'w') as f:
        json.dump(weights_dict, f, indent=2)
    
    print(f"Weights exported to {output_file}")
    return weights_dict


def generate_verilog_from_checkpoint(checkpoint_path='best_model.pth', 
                                      output_va='bsim_nn.va',
                                      output_json='nn_weights.json'):
    """
    从训练好的checkpoint生成Verilog-A模型
    
    参数:
        checkpoint_path: 训练好的模型权重文件 (.pth)
        output_va: 输出Verilog-A文件路径
        output_json: 输出JSON权重文件路径
    """
    print("="*60)
    print("BSIM-NN Verilog-A Model Generator")
    print("="*60)
    
    # 1. 重新创建模型结构
    print("\n[1/4] Loading model structure...")
    model = BSIM_NN_IV(
        input_dim=3,
        hidden_size=32,
        num_hidden_layers=2,
        activation='isru'
    )
    
    # 2. 加载训练好的权重
    print(f"[2/4] Loading weights from {checkpoint_path}...")
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        print("    Model loaded successfully!")
    except FileNotFoundError:
        print(f"    Warning: {checkpoint_path} not found, using uninitialized model")
        print("    Please train the model first using train_nn_model_fixed_v3.py")
        return
    
    model.eval()
    
    # 3. 加载归一化参数（需要从训练数据中获取）
    print("[3/4] Setting normalization parameters...")
    
    # 这些值应该与训练时的值匹配
    # 从训练代码中，这些值是根据训练数据计算的
    # 以下是典型值，实际使用时请从训练脚本中获取
    input_means = np.array([0.5, 0.5, 0.0])    # VGS, VDS, VBS的均值
    input_stds = np.array([0.3, 0.3, 0.1])     # VGS, VDS, VBS的标准差
    
    # 注意：实际值应该从训练数据中获取
    # 训练时可以通过 model.input_means 和 model.input_stds 获取
    if hasattr(model, 'input_means') and model.input_means is not None:
        input_means = model.input_means.numpy()
        input_stds = model.input_stds.numpy()
        print(f"    Using model normalization parameters:")
        print(f"      input_means: {input_means}")
        print(f"      input_stds: {input_stds}")
    else:
        print(f"    WARNING: Using default normalization parameters!")
        print(f"      Please verify these match your training data:")
        print(f"      input_means: {input_means}")
        print(f"      input_stds: {input_stds}")
    
    # 4. 生成Verilog-A
    print("[4/4] Generating Verilog-A code...")
    generator = VerilogAGenerator(model, input_means, input_stds)
    generator.generate(output_va)
    
    # 5. 导出JSON权重（备用）
    export_weights_to_json(model, input_means, input_stds, output_json)
    
    print("\n" + "="*60)
    print("Verilog-A model generation completed!")
    print(f"  Verilog-A file: {output_va}")
    print(f"  JSON weights: {output_json}")
    print("="*60)
    
    # 打印使用说明
    print("\n" + "="*60)
    print("How to use in SPICE/NGSPICE:")
    print("="*60)
    print("1. Compile the Verilog-A model using OpenVAF:")
    print(f"   openvaf {output_va} --ngspice -o bsim_nn.so")
    print("")
    print("2. In NGSPICE, load the model:")
    print("   .control")
    print("   pre_osdi bsim_nn.so")
    print("   .endc")
    print("")
    print("3. Instantiate the transistor:")
    print("   M1 d g 0 0 bsim_nn L=1e-6 W=10e-6")
    print("")
    print("4. Example DC sweep:")
    print("   VGS g 0 DC 0")
    print("   VDS d 0 DC 0")
    print("   .dc VGS 0 1.0 0.01 VDS 0 1.0 0.1")
    print("="*60)


def generate_simple_verilog_example():
    """生成一个简化的Verilog-A示例（用于测试，不依赖训练好的模型）"""
    
    code_lines = [
        '// ============================================================',
        '// BSIM-NN: Simplified Example Verilog-A Model',
        '// ============================================================',
        '',
        '`include "disciplines.vams"',
        '`include "constants.vams"',
        '',
        'module bsim_nn_simple(d, g, s);',
        '    inout d, g, s;',
        '    electrical d, g, s;',
        '',
        '    // Geometry parameters',
        '    parameter real w = 10e-6;',
        '    parameter real l = 1e-6;',
        '',
        '    // Internal variables',
        '    real Vgs, Vds, Ids;',
        '    real h1, h2, h3, h4, h5, h6, h7, h8;',
        '    real h9, h10, h11, h12, h13, h14, h15, h16;',
        '    real log10_Id;',
        '',
        '    // Input normalization parameters',
        '    real mean_vgs = 0.5;',
        '    real scale_vgs = 0.3;',
        '    real mean_vds = 0.5;',
        '    real scale_vds = 0.3;',
        '    real Vgs_n, Vds_n;',
        '',
        '    analog begin',
        '        // Read voltages',
        '        Vgs = V(g, s);',
        '        Vds = V(d, s);',
        '',
        '        // Normalize inputs',
        '        Vgs_n = (Vgs - mean_vgs) / scale_vgs;',
        '        Vds_n = (Vds - mean_vds) / scale_vds;',
        '',
        '        // Hidden layer 1 (ISRU inline: x / sqrt(1 + x^2))',
        '        h1 = ( 0.5*Vgs_n + -0.2*Vds_n + 0.1) / `sqrt(1.0 + (0.5*Vgs_n + -0.2*Vds_n + 0.1)*(0.5*Vgs_n + -0.2*Vds_n + 0.1));',
        '        h2 = ( 0.3*Vgs_n + -0.1*Vds_n + 0.0) / `sqrt(1.0 + (0.3*Vgs_n + -0.1*Vds_n + 0.0)*(0.3*Vgs_n + -0.1*Vds_n + 0.0));',
        '        h3 = (-0.4*Vgs_n +  0.3*Vds_n + -0.1) / `sqrt(1.0 + (-0.4*Vgs_n +  0.3*Vds_n + -0.1)*(-0.4*Vgs_n +  0.3*Vds_n + -0.1));',
        '        h4 = (-0.2*Vgs_n +  0.5*Vds_n + -0.2) / `sqrt(1.0 + (-0.2*Vgs_n +  0.5*Vds_n + -0.2)*(-0.2*Vgs_n +  0.5*Vds_n + -0.2));',
        '        h5 = ( 0.1*Vgs_n + -0.3*Vds_n + 0.2) / `sqrt(1.0 + (0.1*Vgs_n + -0.3*Vds_n + 0.2)*(0.1*Vgs_n + -0.3*Vds_n + 0.2));',
        '        h6 = ( 0.2*Vgs_n +  0.1*Vds_n + -0.1) / `sqrt(1.0 + (0.2*Vgs_n +  0.1*Vds_n + -0.1)*(0.2*Vgs_n +  0.1*Vds_n + -0.1));',
        '        h7 = (-0.1*Vgs_n + -0.2*Vds_n + 0.1) / `sqrt(1.0 + (-0.1*Vgs_n + -0.2*Vds_n + 0.1)*(-0.1*Vgs_n + -0.2*Vds_n + 0.1));',
        '        h8 = ( 0.0*Vgs_n +  0.2*Vds_n + 0.0) / `sqrt(1.0 + (0.0*Vgs_n +  0.2*Vds_n + 0.0)*(0.0*Vgs_n +  0.2*Vds_n + 0.0));',
        '',
        '        // Hidden layer 2 (ISRU inline)',
        '        h9  = ( 0.3*h1 + -0.1*h2 +  0.2*h3 + -0.1*h4 +  0.1*h5 + -0.2*h6 +  0.1*h7 + -0.1*h8 + 0.0) / `sqrt(1.0 + (0.3*h1 + -0.1*h2 +  0.2*h3 + -0.1*h4 +  0.1*h5 + -0.2*h6 +  0.1*h7 + -0.1*h8 + 0.0)*(0.3*h1 + -0.1*h2 +  0.2*h3 + -0.1*h4 +  0.1*h5 + -0.2*h6 +  0.1*h7 + -0.1*h8 + 0.0));',
        '        h10 = ( 0.2*h1 +  0.1*h2 + -0.1*h3 +  0.2*h4 + -0.2*h5 +  0.1*h6 + -0.1*h7 +  0.1*h8 + 0.0) / `sqrt(1.0 + (0.2*h1 +  0.1*h2 + -0.1*h3 +  0.2*h4 + -0.2*h5 +  0.1*h6 + -0.1*h7 +  0.1*h8 + 0.0)*(0.2*h1 +  0.1*h2 + -0.1*h3 +  0.2*h4 + -0.2*h5 +  0.1*h6 + -0.1*h7 +  0.1*h8 + 0.0));',
        '        h11 = (-0.1*h1 +  0.3*h2 +  0.1*h3 + -0.2*h4 +  0.1*h5 +  0.0*h6 + -0.2*h7 +  0.2*h8 + 0.0) / `sqrt(1.0 + (-0.1*h1 +  0.3*h2 +  0.1*h3 + -0.2*h4 +  0.1*h5 +  0.0*h6 + -0.2*h7 +  0.2*h8 + 0.0)*(-0.1*h1 +  0.3*h2 +  0.1*h3 + -0.2*h4 +  0.1*h5 +  0.0*h6 + -0.2*h7 +  0.2*h8 + 0.0));',
        '        h12 = ( 0.1*h1 + -0.2*h2 +  0.0*h3 +  0.3*h4 + -0.1*h5 +  0.2*h6 +  0.1*h7 + -0.2*h8 + 0.0) / `sqrt(1.0 + (0.1*h1 + -0.2*h2 +  0.0*h3 +  0.3*h4 + -0.1*h5 +  0.2*h6 +  0.1*h7 + -0.2*h8 + 0.0)*(0.1*h1 + -0.2*h2 +  0.0*h3 +  0.3*h4 + -0.1*h5 +  0.2*h6 +  0.1*h7 + -0.2*h8 + 0.0));',
        '        h13 = (-0.2*h1 +  0.0*h2 + -0.1*h3 +  0.1*h4 +  0.2*h5 + -0.1*h6 +  0.0*h7 +  0.3*h8 + 0.0) / `sqrt(1.0 + (-0.2*h1 +  0.0*h2 + -0.1*h3 +  0.1*h4 +  0.2*h5 + -0.1*h6 +  0.0*h7 +  0.3*h8 + 0.0)*(-0.2*h1 +  0.0*h2 + -0.1*h3 +  0.1*h4 +  0.2*h5 + -0.1*h6 +  0.0*h7 +  0.3*h8 + 0.0));',
        '        h14 = ( 0.1*h1 +  0.2*h2 + -0.2*h3 +  0.0*h4 + -0.1*h5 +  0.3*h6 +  0.1*h7 + -0.1*h8 + 0.0) / `sqrt(1.0 + (0.1*h1 +  0.2*h2 + -0.2*h3 +  0.0*h4 + -0.1*h5 +  0.3*h6 +  0.1*h7 + -0.1*h8 + 0.0)*(0.1*h1 +  0.2*h2 + -0.2*h3 +  0.0*h4 + -0.1*h5 +  0.3*h6 +  0.1*h7 + -0.1*h8 + 0.0));',
        '        h15 = (-0.1*h1 + -0.1*h2 +  0.1*h3 + -0.1*h4 +  0.2*h5 + -0.2*h6 +  0.3*h7 +  0.0*h8 + 0.0) / `sqrt(1.0 + (-0.1*h1 + -0.1*h2 +  0.1*h3 + -0.1*h4 +  0.2*h5 + -0.2*h6 +  0.3*h7 +  0.0*h8 + 0.0)*(-0.1*h1 + -0.1*h2 +  0.1*h3 + -0.1*h4 +  0.2*h5 + -0.2*h6 +  0.3*h7 +  0.0*h8 + 0.0));',
        '        h16 = ( 0.0*h1 +  0.1*h2 + -0.2*h3 +  0.1*h4 + -0.1*h5 +  0.0*h6 + -0.1*h7 +  0.2*h8 + 0.0) / `sqrt(1.0 + (0.0*h1 +  0.1*h2 + -0.2*h3 +  0.1*h4 + -0.1*h5 +  0.0*h6 + -0.1*h7 +  0.2*h8 + 0.0)*(0.0*h1 +  0.1*h2 + -0.2*h3 +  0.1*h4 + -0.1*h5 +  0.0*h6 + -0.1*h7 +  0.2*h8 + 0.0));',
        '',
        '        // Output layer',
        '        log10_Id = -6.0',
        '            + 0.2*h9  + 0.1*h10 + -0.1*h11 + 0.2*h12',
        '            + -0.1*h13 + 0.1*h14 + -0.2*h15 + 0.1*h16;',
        '',
        '        // Clamp and convert to linear current',
        '        log10_Id = (log10_Id < -8) ? -8 : ((log10_Id > -3) ? -3 : log10_Id);',
        '        Ids = pow(10, log10_Id) * (w / 10e-6);',
        '        if (Ids < 1e-12) Ids = 1e-12;',
        '        if (Ids > 1e-3) Ids = 1e-3;',
        '        if (Vds == 0) Ids = 0;',
        '',
        '        // Output current',
        '        I(d, s) <+ Ids;',
        '        I(g, s) <+ 0;',
        '    end',
        '',
        'endmodule'
    ]
    
    with open('bsim_nn_simple.va', 'w') as f:
        f.write('\n'.join(code_lines))
    
    print("Simple Verilog-A example generated: bsim_nn_simple.va")
    return 'bsim_nn_simple.va'


if __name__ == "__main__":
    # 生成完整的Verilog-A模型（需要训练好的checkpoint）
    generate_verilog_from_checkpoint(
        checkpoint_path='best_model.pth',  # 或 'nn_model_final.pth'
        output_va='bsim_nn.va',
        output_json='nn_weights.json'
    )
    
    # 同时生成一个简化示例（不需要训练好的模型）
    generate_simple_verilog_example()
    
    print("\n" + "="*60)
    print("To compile the Verilog-A model with OpenVAF:")
    print("  openvaf bsim_nn.va --ngspice -o bsim_nn.so")
    print("")
    print("To test in NGSPICE:")
    print("  ngspice -b test_bsim_nn.sp")
    print("="*60)