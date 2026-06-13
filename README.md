# NN_device_model -- Neural Network Compact Model for Semiconductor Devices

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Neural network-based semiconductor device compact model. Trains a PyTorch MLP to fit BSIM-generated IV characteristics, exports weights to Verilog-A, compiles to OSDI via OpenVAF, and runs circuit-level simulation in ngspice.

## Project Overview

Traditional semiconductor compact models (BSIM3/BSIM4) rely on complex physics-based analytical equations with lengthy parameter extraction cycles. This project explores using **feedforward neural networks (MLPs)** as compact models, enabling an end-to-end flow from device IV data to simulatable circuit models.

### Workflow

```
BSIM Physics Model --> Generate IV Training Data --> PyTorch NN Training --> Export Verilog-A
                                                                                |
                                                    ngspice Circuit Sim <-- OpenVAF Compile to OSDI
```

## Directory Structure

| Directory | Description |
|-----------|-------------|
| `version1/` | Proof of concept: basic ANN training + first Verilog-A export attempts |
| `version2/` | Improved: weighted loss function for better subthreshold fitting |
| `version3/` | BSIM3 IV data generator, 3D surface visualization, multi-input MLP |
| `version4/` | Paper-grade: ISRU activation, derivative-assisted loss (gm/gds), compact model architecture |
| `version5/` | **Current working version**: simplified 3-16-16-1 Tanh network with numerical stability fixes |
| `ngspice_simulation/` | ngspice test circuits (.cir/.sp files) and compiled .osdi models |

## Version Evolution

### Version 1 -- Proof of Concept
- 2 inputs (VGS, VDS) -> 10 neurons (Tanh) -> 1 output
- Training data from simplified MOSFET equations
- Multiple Verilog-A export script attempts

### Version 2 -- Weighted Training
- Increased loss weight for low-current (subthreshold) regions
- Improved fitting accuracy in weak inversion

### Version 3 -- BSIM3 Data Source
- BSIM3 physics model as training data generator
- 5 inputs (VGS, VDS, L, HFIN, EOT) -> 16->16 -> 3 outputs (ID, IG_POS, IG_NEG)
- Supports device geometry and process parameter variability
- 3D surface plots for IV characteristics

### Version 4 -- Paper Implementation
- Based on Tung & Hu, IEEE TED 2023/2024 NN compact model framework
- **ISRU** activation: `f(x) = x / sqrt(1 + x^2)` -- no exp(), Verilog-A friendly
- **Derivative-assisted loss**: simultaneously fits ID, gm (transconductance), and gds (output conductance)
- `ID = VDS * exp(y1)` output transformation ensures ID=0 at VDS=0

### Version 5 -- Current Working Version (latest)
- **3->16->16->1** Tanh MLP, outputting log10(ID)
- Simplified but convergent training pipeline
- Direct Verilog-A code generation with extracted weights
- **Known issues**: OSDI simulation hangs (see below)

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Generation | Python + NumPy (BSIM3 analytical model) |
| NN Training | PyTorch (MLP, MSE loss, Adam optimizer) |
| Model Export | Custom Python scripts -> Verilog-A |
| VA Compilation | OpenVAF -> OSDI shared library |
| Circuit Simulation | ngspice with OSDI support |

## Dependencies

```bash
# Python packages
pip install torch numpy pandas matplotlib tqdm scikit-learn

# OpenVAF (build from source or use precompiled binary)
# https://github.com/OpenVAF/OpenVAF

# ngspice (must be built with OSDI support)
# https://ngspice.sourceforge.io/
```

## Quick Start

### 1. Train the NN Model

```bash
cd version5
python working_nn_model.py
# Output: working_model.pth + working_bsim_nn.va
```

### 2. Compile Verilog-A to OSDI

```bash
openvaf working_bsim_nn.va --ngspice -o bsim_nn.osdi
```

### 3. Run ngspice Simulation

```spice
* test.cir
.control
pre_osdi bsim_nn.osdi
.endc

.model NMOS1 bsim_nn(w=10e-6 l=1e-6)
M1 d g 0 0 NMOS1
Vds d 0 1.0
Vgs g 0 0.7

.op
.control
run
print i(vds)
.endc
.end
```

## Known Issues & Future Work

### 1. OSDI Simulation Hangs

The Verilog-A model has several issues that cause SPICE Newton-Raphson solver divergence:

| Issue | Location | Impact |
|-------|----------|--------|
| `if (VDS == 0)` floating-point equality | Output section | Discontinuous Jacobian, solver non-convergence |
| `std_vbs = 0.000001` near-zero divisor | Input normalization | Overflow when VBS != 0 |
| Large 2nd-layer weights (e.g., -7.66) | Hidden layer 2 | tanh saturation -> near-zero derivatives -> singular Jacobian |
| Ternary operator `? :` clipping | Clamp section | Non-differentiable transitions |
| No gate/body branch currents | Terminal currents | Potentially singular Jacobian matrix |

### 2. NN Model Accuracy

The current simplified BSIM-derived training data does not accurately represent real process behavior. A 3-input MLP cannot fully capture:
- Subthreshold exponential region
- Linear/triode region square-law behavior
- Saturation region with channel-length modulation
- Smooth transitions between operating regions

**Recommended directions:**
- Generate training data from actual BSIM4 models via ngspice batch simulation
- Increase network depth/width with mixed activation strategies suited to different operating regions
- Add physics-informed constraints to the loss function

## References

- Tung, C.T. & Hu, C. "Neural Network-Based BSIM Transistor Model Framework", IEEE Transactions on Electron Devices, 2023
- BSIM3v3 Manual, UC Berkeley Device Group
- OpenVAF: <https://github.com/OpenVAF/OpenVAF>
- ngspice: <https://ngspice.sourceforge.io/>

---

*Project by Guo-4869 | NN Device Modeling & Simulation*
