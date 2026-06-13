import numpy as np
import torch


class VerilogAExporter:
    def __init__(self, model_path='ann_bsim_model.pth'):
        checkpoint = torch.load(model_path, map_location='cpu')
        self.scaler_X = checkpoint['scaler_X']
        self.scaler_y = checkpoint['scaler_y']
        self.weights = checkpoint['weights']
        self.biases = checkpoint['biases']
        self.hidden_sizes = checkpoint['hidden_sizes']
        
    def generate_verilog_a(self, output_file='ann_bsim.va'):
        """生成Verilog-A模型代码"""
        
        mean_x = self.scaler_X.mean_
        scale_x = self.scaler_X.scale_
        mean_y = self.scaler_y.mean_[0]
        scale_y = self.scaler_y.scale_[0]
        
        va_code = f'''// ANN-based MOSFET Model Trained on BSIM Data
// Inputs: Vgs, Vds, log10(W), log10(L), Temp/100
// Output: log10(Ids)

`include "disciplines.vams"

module ann_bsim(d, g, s, b);
    inout d, g, s, b;
    electrical d, g, s, b;
    
    parameter real w = 10e-6;
    parameter real l = 1e-6;
    
    // Normalization parameters
    real mean_vgs = {mean_x[0]:.8e};
    real mean_vds = {mean_x[1]:.8e};
    real mean_logw = {mean_x[2]:.8e};
    real mean_logl = {mean_x[3]:.8e};
    real mean_temp = {mean_x[4]:.8e};
    
    real scale_vgs = {scale_x[0]:.8e};
    real scale_vds = {scale_x[1]:.8e};
    real scale_logw = {scale_x[2]:.8e};
    real scale_logl = {scale_x[3]:.8e};
    real scale_temp = {scale_x[4]:.8e};
    
    real mean_logids = {mean_y:.8e};
    real scale_logids = {scale_y:.8e};
    
    // Layer 1 weights and biases
    real w1[{self.weights[0].shape[0]}][{self.weights[0].shape[1]}] = '{{'''
        for i in range(self.weights[0].shape[0]):
            if i > 0:
                va_code += ',\n                                 '
            va_code += '{' + ', '.join([f"{x:.8e}" for x in self.weights[0][i]]) + '}'
        va_code += '};'
        
        va_code += f'''
    real b1[{self.biases[0].shape[0]}] = '{{{', '.join([f"{x:.8e}" for x in self.biases[0]])}}}';
    
    // Layer 2 weights and biases
    real w2[{self.weights[1].shape[0]}][{self.weights[1].shape[1]}] = '{{'''
        for i in range(self.weights[1].shape[0]):
            if i > 0:
                va_code += ',\n                                 '
            va_code += '{' + ', '.join([f"{x:.8e}" for x in self.weights[1][i]]) + '}'
        va_code += '};'
        
        va_code += f'''
    real b2[{self.biases[1].shape[0]}] = '{{{', '.join([f"{x:.8e}" for x in self.biases[1]])}}}';
    
    // Layer 3 weights and biases
    real w3[{self.weights[2].shape[0]}][{self.weights[2].shape[1]}] = '{{'''
        for i in range(self.weights[2].shape[0]):
            if i > 0:
                va_code += ',\n                                 '
            va_code += '{' + ', '.join([f"{x:.8e}" for x in self.weights[2][i]]) + '}'
        va_code += '};'
        
        va_code += f'''
    real b3[{self.biases[2].shape[0]}] = '{{{', '.join([f"{x:.8e}" for x in self.biases[2]])}}}';
    
    // Output layer
    real w4[{self.weights[3].shape[0]}][{self.weights[3].shape[1]}] = '{{'''
        for i in range(self.weights[3].shape[0]):
            if i > 0:
                va_code += ',\n                                 '
            va_code += '{' + ', '.join([f"{x:.8e}" for x in self.weights[3][i]]) + '}'
        va_code += '};'
        
        va_code += f'''
    real b4[{self.biases[3].shape[0]}] = '{{{', '.join([f"{x:.8e}" for x in self.biases[3]])}}}';
    
    // Forward propagation
    real function forward(real vgs, real vds, real w, real l, real temp);
        real x[5];
        real h1[32], h2[32], h3[32];
        real logw, logl;
        integer i, j;
    begin
        logw = `ln(w) / `ln(10);
        logl = `ln(l) / `ln(10);
        
        x[0] = (vgs - mean_vgs) / scale_vgs;
        x[1] = (vds - mean_vds) / scale_vds;
        x[2] = (logw - mean_logw) / scale_logw;
        x[3] = (logl - mean_logl) / scale_logl;
        x[4] = (temp - mean_temp) / scale_temp;
        
        // Hidden layer 1
        for (i = 0; i < 32; i = i + 1) begin
            h1[i] = b1[i];
            for (j = 0; j < 5; j = j + 1)
                h1[i] = h1[i] + w1[i][j] * x[j];
            h1[i] = tanh(h1[i]);
        end
        
        // Hidden layer 2
        for (i = 0; i < 32; i = i + 1) begin
            h2[i] = b2[i];
            for (j = 0; j < 32; j = j + 1)
                h2[i] = h2[i] + w2[i][j] * h1[j];
            h2[i] = tanh(h2[i]);
        end
        
        // Hidden layer 3
        for (i = 0; i < 32; i = i + 1) begin
            h3[i] = b3[i];
            for (j = 0; j < 32; j = j + 1)
                h3[i] = h3[i] + w3[i][j] * h2[j];
            h3[i] = tanh(h3[i]);
        end
        
        // Output layer
        forward = b4[0];
        for (j = 0; j < 32; j = j + 1)
            forward = forward + w4[0][j] * h3[j];
        forward = forward * scale_logids + mean_logids;
    end
    endfunction
    
    real vgs, vds, logids, ids;
    
    analog begin
        vgs = V(g, s);
        vds = V(d, s);
        
        logids = forward(vgs, vds, w, l, 27.0);
        ids = pow(10, logids);
        
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
        
        return va_code


if __name__ == "__main__":
    exporter = VerilogAExporter('ann_bsim_model.pth')
    exporter.generate_verilog_a('ann_bsim.va')