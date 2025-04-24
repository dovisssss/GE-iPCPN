import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
# 参数设置
m = 1.0  # 质量 (kg)
c = 40.0  # 阻尼系数
k = 3000.0  # 线性刚度
kc = 5e8  # 立方刚度
fs = 2000  # 采样频率 (Hz)
dt = 1 / fs  # 时间间隔 (s)
T = 40  # 总时间 (s)
amplitude_target = 20  # 目标幅值 (N)
t = torch.arange(0, T, dt)  # 时间向量
N = len(t)  # 采样点数

# 多正弦激励信号
M = 150  # 谐波数量
f_min = 5  # 最小频率 (Hz)
f_max = 80  # 最大频率 (Hz)
frequencies = torch.linspace(f_min, f_max, M)  # 生成M个频率
amplitudes = torch.ones(M)  # 每个频率的振幅
phases = 2 * np.pi * torch.rand(M)  # 随机相位

def min_max_normalize(data):
    data_min = torch.min(data)
    data_max = torch.max(data)
    normalized_data = 2 * (data - data_min) / (data_max - data_min) - 1
    return normalized_data

u = torch.zeros(N)
for i in range(M):
    u += amplitudes[i] * torch.sin(2 * np.pi * frequencies[i] * t + phases[i])
# 调整幅值到目标值
current_max_amplitude = torch.max(torch.abs(u)).item()
u_scaled = u * amplitude_target / current_max_amplitude

excitation_max = torch.max(u_scaled).item()  # 激励信号的最大值
u_norm= min_max_normalize(u_scaled) #归一化后的激励信号

# 初始条件
x0 = 0.0  # 初始位移
v0 = 0.0  # 初始速度

# Duffing系统的微分方程
def duffing_system(x, v, u):
    dxdt = v
    dvdt = (-c * v - k * x - kc * x**3 + u) / m
    return dxdt, dvdt

# 龙格库塔方法
def runge_kutta(x, v, u, dt):
    k1_x, k1_v = duffing_system(x, v, u)
    k2_x, k2_v = duffing_system(x + k1_x * dt / 2, v + k1_v * dt / 2, u)
    k3_x, k3_v = duffing_system(x + k2_x * dt / 2, v + k2_v * dt / 2, u)
    k4_x, k4_v = duffing_system(x + k3_x * dt, v + k3_v * dt, u)
    x_new = x + (k1_x + 2 * k2_x + 2 * k3_x + k4_x) * dt / 6
    v_new = v + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) * dt / 6
    return x_new, v_new

# 初始化位移和速度
x = torch.zeros(N)
v = torch.zeros(N)
x[0] = x0
v[0] = v0

# 使用龙格库塔方法求解
for i in range(1, N):
    x[i], v[i] = runge_kutta(x[i - 1], v[i - 1], u[i - 1], dt)

# 添加噪声
noise_level = 0.05  # 噪声水平

x_n = np.random.normal(0,1,N)
x_n = torch.from_numpy(x_n / np.std(x_n))
x_noisy = (x + noise_level * torch.std(x) * x_n)
v_n = np.random.normal(0,1,N)
v_n = torch.from_numpy(v_n / np.std(v_n))
v_noisy = (v + noise_level * torch.std(v) * v_n)


# 计算噪声数据的最大值
displacement_noisy_max = torch.max(torch.abs(x_noisy)).item()
velocity_noisy_max = torch.max(torch.abs(v_noisy)).item()
x_norm = min_max_normalize(x)
v_norm = min_max_normalize(v)
x_noisy_norm = min_max_normalize(x_noisy)
v_noisy_norm = min_max_normalize(v_noisy)

# 保存为.mat文件
data = {
    "fs": fs,
    "excitation": u_norm.numpy(),
    "excitation_max": excitation_max,
    "displacement": x_norm.numpy(),
    "displacement_noisy": x_noisy_norm.numpy(),
    "displacement_noisy_max": displacement_noisy_max,
    "velocity": v_norm.numpy(),
    "velocity_noisy": v_noisy_norm.numpy(),
    "velocity_noisy_max": velocity_noisy_max,
}

sio.savemat("duffing_simulation_noise5.mat", data)
print("数据已保存为 duffing_simulation_noise5.mat")

# 激励信号
plt.subplot(3, 1, 1)
plt.plot(t, u_norm, label="Excitation Signal")
plt.title("Excitation Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

# 位移响应
plt.subplot(3, 1, 2)
plt.plot(t, x_norm, label="Displacement Response", color="orange")
plt.plot(t, x_noisy_norm, label="Noisy Displacement Response", color="red", alpha=0.5)
plt.title("Displacement Response")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

# 速度响应
plt.subplot(3, 1, 3)
plt.plot(t, v_norm, label="Velocity Response", color="green")
plt.plot(t, v_noisy_norm, label="Noisy Velocity Response", color="blue", alpha=0.5)
plt.title("Velocity Response")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
