# 读取数据并绘图
import numpy as np
import matplotlib.pyplot as plt

# 读取转移特性数据
data = np.loadtxt('transfer_compare.dat')
vgs = data[:, 0]
id_nn = data[:, 1]
id_bsim = data[:, 2]

plt.figure()
plt.semilogy(vgs, id_nn, label='NN Model')
plt.semilogy(vgs, id_bsim, label='BSIM Model')
plt.xlabel('VGS (V)')
plt.ylabel('ID (A)')
plt.legend()
plt.grid(True)
plt.show()