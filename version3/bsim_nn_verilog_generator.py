# bsim_nn_verilog_generator.py
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BSIMNNConfig:
    """BSIM-NN模型配置"""
    # 网络结构
    hidden_size: int = 32  # 隐藏层神经元数量
    num_hidden_layers: int = 3  # 隐藏层数量
    input_size: int = 5  # 输入: Vgs, Vds, L, W, Temp
    
    # 归一化参数 (从训练数据中获取)
    mean_vgs: float = 0.0
    scale_vgs: float = 0.5
    mean_vds: float = 0.0
    scale_vds: float = 0.5
    mean_l: float = 18e-9
    scale_l: float = 4e-9
    mean_w: float = 1e-6
    scale_w: float = 0.5e-6
    mean_temp: float = 27.0
    scale_temp: float = 50.0
    
    # 输出归一化
    mean_id: float = 0.0
    scale_id: float = 1.0
    
    # 物理约束
    min_id: float = 1e-12
    max_id: float = 0.01
    
    # 模型版本
    version: str = "1.0"
    model_name: str = "bsim_nn"


class BSIMNNVerilogGenerator:
    """BSIM-NN Verilog-A模型生成器 (ngspice兼容格式)"""
    
    def __init__(self, config: BSIMNNConfig):
        self.config = config
        self.weights = None
        self.code_lines = []
        
    def load_weights(self, weights_file: str):
        """从JSON文件加载训练好的权重"""
        with open(weights_file, 'r') as f:
            self.weights = json.load(f)
        print(f"Weights loaded from {weights_file}")
        return self.weights
    
    def load_weights_from_pytorch(self, pytorch_model_path: str):
        """从PyTorch模型文件加载权重"""
        import torch
        state_dict = torch.load(pytorch_model_path, map_location='cpu')
        
        weights = {}
        for name, param in state_dict.items():
            if 'weight' in name:
                weights[name] = param.numpy().tolist()
            elif 'bias' in name:
                weights[name] = param.numpy().tolist()
        
        self.weights = weights
        return weights
    
    def generate(self, output_file: str) -> str:
        """生成完整的Verilog-A模型代码"""
        
        self._write_header()
        self._write_parameters()
        self._write_internal_variables()
        self._write_normalization()
        
        if self.weights:
            self._write_neural_network_with_weights()
        else:
            self._write_neural_network_template()
        
        self._write_output_scaling()
        self._write_physical_constraints()
        self._write_analog_block()
        self._write_footer()
        
        # 写入文件
        with open(output_file, 'w') as f:
            f.write('\n'.join(self.code_lines))
        
        print(f"Verilog-A model generated: {output_file}")
        return output_file
    
    def _write_header(self):
        """写入文件头"""
        self.code_lines.extend([
            "// ============================================================",
            f"// BSIM-NN: Neural Network Based MOSFET Compact Model v{self.config.version}",
            "// ============================================================",
            "// Reference: C.T. Tung and C. Hu, IEEE Transactions on Electron Devices, 2023",
            "//           \"Neural Network-Based BSIM Transistor Model Framework\"",
            "// ============================================================",
            "// Generated from PyTorch training",
            f"// Hidden layers: {self.config.num_hidden_layers} x {self.config.hidden_size} neurons",
            "// ============================================================",
            "",
            "`include \"disciplines.vams\"",
            "",
        ])
        
        # 模块定义
        self.code_lines.extend([
            f"module {self.config.model_name}(d, g, s, b);",
            "    inout d, g, s, b;",
            "    electrical d, g, s, b;",
            "",
        ])
    
    def _write_parameters(self):
        """写入模型参数"""
        self.code_lines.extend([
            "    // ==============================================",
            "    // Geometry and Physical Parameters",
            "    // ==============================================",
            f"    parameter real w = {self.config.mean_w:.4e};  // Channel width (m)",
            f"    parameter real l = {self.config.mean_l:.4e};  // Channel length (m)",
            "    parameter real nfin = 1;        // Number of fins",
            "    parameter real hfin = 46e-9;     // Fin height (m)",
            "    parameter real eot = 0.78e-9;    // Equivalent oxide thickness (m)",
            "    parameter real temp = 27;        // Temperature (Celsius)",
            "",
            "    // Model tuning parameters",
            "    parameter real alpha = 0.5;       // Smoothing parameter for ID",
            "    parameter real delta = 1e-6;      // Smoothing parameter for IG",
            "    parameter real i0 = 1e-12;        // Offset current",
            "",
        ])
    
    def _write_internal_variables(self):
        """写入内部变量定义"""
        # 生成隐藏层变量
        hidden_vars = [f"h{i+1}" for i in range(self.config.hidden_size)]
        
        self.code_lines.extend([
            "    // ==============================================",
            "    // Internal Variables",
            "    // ==============================================",
            "    real Vgs, Vds, Vbs;",
            "    real Ids, Igs, Ibs;",
            "    real gm, gds;",
            "",
            "    // Normalized inputs",
            "    real Vgs_n, Vds_n, L_n, W_n, Temp_n;",
            "",
            "    // Hidden layer neurons",
            f"    real {', '.join(hidden_vars)};",
            "",
            "    // Output",
            "    real logIds;",
            "",
        ])
    
    def _write_normalization(self):
        """写入输入归一化"""
        self.code_lines.extend([
            "    // ==============================================",
            "    // Input Normalization (Z-score)",
            "    // ==============================================",
            f"    real mean_vgs = {self.config.mean_vgs};",
            f"    real scale_vgs = {self.config.scale_vgs};",
            f"    real mean_vds = {self.config.mean_vds};",
            f"    real scale_vds = {self.config.scale_vds};",
            f"    real mean_l = {self.config.mean_l:.4e};",
            f"    real scale_l = {self.config.scale_l:.4e};",
            f"    real mean_w = {self.config.mean_w:.4e};",
            f"    real scale_w = {self.config.scale_w:.4e};",
            f"    real mean_temp = {self.config.mean_temp};",
            f"    real scale_temp = {self.config.scale_temp};",
            "",
        ])
    
    def _write_neural_network_with_weights(self):
        """使用实际训练权重写入神经网络"""
        
        self.code_lines.extend([
            "    // ==============================================",
            "    // Neural Network Forward Pass",
            "    // ==============================================",
            "",
            "    // Normalize inputs",
            "    Vgs_n = (Vgs - mean_vgs) / scale_vgs;",
            "    Vds_n = (Vds - mean_vds) / scale_vds;",
            "    L_n = (l - mean_l) / scale_l;",
            "    W_n = (w - mean_w) / scale_w;",
            "    Temp_n = (temp - mean_temp) / scale_temp;",
            "",
        ])
        
        # 生成每一层的权重计算
        self._generate_layer_computation()
        
        # 输出层
        self._generate_output_layer()
    
    def _generate_layer_computation(self):
        """生成隐藏层计算代码"""
        
        # 获取权重
        weights_l1 = self.weights.get('layer1.weight', [])
        bias_l1 = self.weights.get('layer1.bias', [])
        
        if weights_l1 and len(weights_l1) > 0:
            # 第一层: 5个输入 -> hidden_size个神经元
            for i in range(self.config.hidden_size):
                # 构建表达式
                expr = f"h{i+1} = tanh("
                
                # 添加偏置
                bias_val = bias_l1[i] if i < len(bias_l1) else 0
                expr += f"{bias_val:.10f}"
                
                # 添加权重项
                if i < len(weights_l1[0]) if weights_l1 else False:
                    for j, input_name in enumerate(['Vgs_n', 'Vds_n', 'L_n', 'W_n', 'Temp_n']):
                        if j < len(weights_l1):
                            w_val = weights_l1[j][i] if len(weights_l1[j]) > i else 0
                            expr += f" + {w_val:.10f}*{input_name}"
                
                expr += ");"
                self.code_lines.append(f"        {expr}")
        else:
            # 如果没有权重，使用示例权重
            self._write_example_weights()
    
    def _write_example_weights(self):
        """写入示例权重（用于测试）"""
        
        # 第一隐藏层 (5 -> hidden_size)
        self.code_lines.append("        // Layer 1: Input -> Hidden")
        for i in range(min(8, self.config.hidden_size)):
            # 使用随机值作为示例
            w1 = np.random.randn(5) * 0.5
            b1 = np.random.randn(1) * 0.1
            expr = f"h{i+1} = tanh({b1[0]:.6f}"
            for j, input_name in enumerate(['Vgs_n', 'Vds_n', 'L_n', 'W_n', 'Temp_n']):
                expr += f" + {w1[j]:.6f}*{input_name}"
            expr += ");"
            self.code_lines.append(f"        {expr}")
        
        # 如果隐藏层大于8，继续生成
        for i in range(8, self.config.hidden_size):
            self.code_lines.append(f"        h{i+1} = tanh(0.0);  // Placeholder")
        
        # 第二隐藏层 (hidden_size -> hidden_size)
        if self.config.num_hidden_layers >= 2:
            self.code_lines.append("")
            self.code_lines.append("        // Layer 2: Hidden -> Hidden")
            # 简化实现，实际应从权重文件读取
            for i in range(min(8, self.config.hidden_size)):
                expr = f"h{i+1} = tanh(0.5*h{i+1}"
                for j in range(min(8, self.config.hidden_size)):
                    if j != i:
                        expr += f" + 0.1*h{j+1}"
                expr += ");"
                self.code_lines.append(f"        {expr}")
        
        # 第三隐藏层
        if self.config.num_hidden_layers >= 3:
            self.code_lines.append("")
            self.code_lines.append("        // Layer 3: Hidden -> Hidden")
            for i in range(min(8, self.config.hidden_size)):
                expr = f"h{i+1} = tanh(0.5*h{i+1}"
                for j in range(min(8, self.config.hidden_size)):
                    if j != i:
                        expr += f" + 0.05*h{j+1}"
                expr += ");"
                self.code_lines.append(f"        {expr}")
    
    def _generate_output_layer(self):
        """生成输出层计算"""
        
        out_weights = self.weights.get('output.weight', [])
        out_bias = self.weights.get('output.bias', 0)
        
        if out_weights and len(out_weights) > 0:
            expr = f"logIds = {out_bias:.10f}"
            for i in range(min(self.config.hidden_size, len(out_weights))):
                w_val = out_weights[i] if isinstance(out_weights, list) and i < len(out_weights) else 0
                expr += f" + {w_val:.10f}*h{i+1}"
            expr += ";"
            self.code_lines.append(f"        {expr}")
        else:
            # 示例输出层
            expr = "logIds = -1.0"
            for i in range(min(8, self.config.hidden_size)):
                expr += f" + 0.5*h{i+1}"
            expr += ";"
            self.code_lines.append(f"        {expr}")
    
    def _write_neural_network_template(self):
        """写入神经网络模板（无预训练权重）"""
        
        self.code_lines.extend([
            "    // ==============================================",
            "    // Neural Network Forward Pass (Template)",
            "    // ==============================================",
            "    // NOTE: Replace with actual trained weights",
            "",
            "    // Normalize inputs",
            "    Vgs_n = (Vgs - mean_vgs) / scale_vgs;",
            "    Vds_n = (Vds - mean_vds) / scale_vds;",
            "    L_n = (l - mean_l) / scale_l;",
            "    W_n = (w - mean_w) / scale_w;",
            "    Temp_n = (temp - mean_temp) / scale_temp;",
            "",
        ])
        
        # 生成随机初始化的示例权重
        np.random.seed(42)
        
        # 第一层
        self.code_lines.append("        // Layer 1: 5 inputs -> 32 neurons")
        for i in range(self.config.hidden_size):
            weights = np.random.randn(5) * 0.5
            bias = np.random.randn(1) * 0.1
            expr = f"h{i+1} = tanh({bias[0]:.6f}"
            for j, input_name in enumerate(['Vgs_n', 'Vds_n', 'L_n', 'W_n', 'Temp_n']):
                expr += f" + {weights[j]:.6f}*{input_name}"
            expr += ");"
            self.code_lines.append(f"        {expr}")
        
        self.code_lines.append("")
        self.code_lines.append("        // Output layer: 32 -> 1")
        out_weights = np.random.randn(self.config.hidden_size) * 0.5
        out_bias = -1.0
        expr = f"logIds = {out_bias:.6f}"
        for i in range(self.config.hidden_size):
            expr += f" + {out_weights[i]:.6f}*h{i+1}"
        expr += ";"
        self.code_lines.append(f"        {expr}")
        
        self.code_lines.append("")
        self.code_lines.append("        // WARNING: Using random weights! Train model for accuracy.")
    
    def _write_output_scaling(self):
        """写入输出反归一化"""
        self.code_lines.extend([
            "",
            "    // ==============================================",
            "    // Output De-normalization",
            "    // ==============================================",
            "    // Convert from log domain to linear current",
            f"    Ids = pow(10, logIds);",
            "",
            "    // Apply geometry scaling",
            "    Ids = Ids * (w / l);",
            "",
        ])
    
    def _write_physical_constraints(self):
        """写入物理约束"""
        self.code_lines.extend([
            "    // ==============================================",
            "    // Physical Constraints",
            "    // ==============================================",
            f"    if (Ids < {self.config.min_id}) Ids = {self.config.min_id};",
            f"    if (Ids > {self.config.max_id}) Ids = {self.config.max_id};",
            "",
            "    // Ensure ID=0 at VDS=0 (Ohmic contact)",
            "    if (Vds == 0) Ids = 0;",
            "",
            "    // Gate current (negligible for DC simulation)",
            "    Igs = 0;",
            "    Ibs = 0;",
            "",
            "    // Conductances for AC simulation",
            "    gm = ddx(Ids, Vgs);",
            "    gds = ddx(Ids, Vds);",
            "",
        ])
    
    def _write_analog_block(self):
        """写入analog块"""
        self.code_lines.extend([
            "    // ==============================================",
            "    // Analog Behavior Block",
            "    // ==============================================",
            "    analog begin",
            "        // Read terminal voltages",
            "        Vgs = V(g, s);",
            "        Vds = V(d, s);",
            "        Vbs = V(b, s);",
            "        ",
            "        // Calculate currents (from neural network above)",
            "        ",
            "        // Assign currents to terminals",
            "        I(d, s) <+ Ids;",
            "        I(g, s) <+ Igs;",
            "        I(b, s) <+ Ibs;",
            "    end",
            "",
        ])
    
    def _write_footer(self):
        """写入文件结尾"""
        self.code_lines.extend([
            "endmodule",
            "",
            "// ============================================================",
            "// End of BSIM-NN Verilog-A Model",
            "// ============================================================",
        ])


class WeightConverter:
    """权重转换器：将PyTorch模型转换为Verilog-A格式"""
    
    @staticmethod
    def convert_pytorch_to_json(model_path: str, output_json: str, config: BSIMNNConfig = None):
        """转换PyTorch模型为JSON权重文件"""
        import torch
        
        # 加载模型
        state_dict = torch.load(model_path, map_location='cpu')
        
        weights_dict = {}
        
        # 提取所有权重和偏置
        for name, param in state_dict.items():
            if 'weight' in name:
                # 转换为列表格式
                weights_dict[name] = param.numpy().tolist()
            elif 'bias' in name:
                weights_dict[name] = param.numpy().tolist()
        
        # 保存JSON
        with open(output_json, 'w') as f:
            json.dump(weights_dict, f, indent=2)
        
        print(f"Weights saved to {output_json}")
        
        # 如果提供了配置，更新归一化参数
        if config:
            # 从模型统计中提取归一化参数
            # 这里需要根据实际训练时的统计值设置
            pass
        
        return weights_dict
    
    @staticmethod
    def generate_verilog_from_json(weights_json: str, output_va: str, config: BSIMNNConfig = None):
        """从JSON权重文件生成Verilog-A模型"""
        
        if config is None:
            config = BSIMNNConfig()
        
        generator = BSIMNNVerilogGenerator(config)
        generator.load_weights(weights_json)
        generator.generate(output_va)
        
        return output_va


# ============================================================
# 使用示例
# ============================================================

def example_generate_simple_model():
    """生成简单示例模型（用于测试）"""
    
    config = BSIMNNConfig(
        hidden_size=8,
        num_hidden_layers=1,
        mean_vgs=1.5,
        scale_vgs=0.87,
        mean_vds=1.5,
        scale_vds=0.86,
        mean_l=1e-6,
        scale_l=0.5e-6,
        mean_w=10e-6,
        scale_w=5e-6,
        min_id=1e-12,
        max_id=0.01,
        model_name="bsim_nn_test"
    )
    
    generator = BSIMNNVerilogGenerator(config)
    generator.generate("bsim_nn_test.va")
    
    print("Test model generated: bsim_nn_test.va")


def example_generate_from_trained_model():
    """从训练好的模型生成Verilog-A"""
    
    config = BSIMNNConfig(
        hidden_size=32,
        num_hidden_layers=3,
        mean_vgs=0.0,
        scale_vgs=0.5,
        mean_vds=0.0,
        scale_vds=0.5,
        mean_l=18e-9,
        scale_l=4e-9,
        mean_w=1e-6,
        scale_w=0.5e-6,
        mean_temp=27.0,
        scale_temp=50.0,
        min_id=1e-12,
        max_id=0.01,
        model_name="bsim_nn_advanced"
    )
    
    # 方式1: 从JSON加载权重
    generator = BSIMNNVerilogGenerator(config)
    generator.load_weights("trained_weights.json")
    generator.generate("bsim_nn_advanced.va")
    
    # 方式2: 直接从PyTorch模型转换
    # WeightConverter.convert_pytorch_to_json("bsim_nn_model.pth", "trained_weights.json", config)
    # WeightConverter.generate_verilog_from_json("trained_weights.json", "bsim_nn_advanced.va", config)


if __name__ == "__main__":
    # 生成测试模型
    example_generate_simple_model()
    
    # 如果需要从训练模型生成，取消下面的注释
    # example_generate_from_trained_model()