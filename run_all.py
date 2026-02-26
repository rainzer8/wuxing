"""
润泽-制化方程数值模拟脚本
生成论文图1、图2、图3
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os

# 创建保存图片的文件夹
os.makedirs('images', exist_ok=True)

# 设置中文字体（可选，若无中文字体可注释掉）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def wuxing_model(X, t, alpha, beta, gamma, d):
    """五行微分方程组"""
    x1, x2, x3, x4, x5 = X
    dx1 = d + alpha * x5 - beta * x4 - gamma * x1
    dx2 = d + alpha * x1 - beta * x5 - gamma * x2
    dx3 = d + alpha * x2 - beta * x1 - gamma * x3
    dx4 = d + alpha * x3 - beta * x2 - gamma * x4
    dx5 = d + alpha * x4 - beta * x3 - gamma * x5
    return [dx1, dx2, dx3, dx4, dx5]


# ==================== 图1：健康状态稳定回归 ====================
print("生成图1...")
alpha1, beta1, gamma1, d1 = 0.2, 0.3, 0.5, 1.0
x_star1 = d1 / (gamma1 - alpha1 + beta1)

t = np.linspace(0, 50, 1000)
X0_1 = [x_star1, x_star1, x_star1 + 0.2, x_star1, x_star1]  # 肝+0.2

sol1 = odeint(wuxing_model, X0_1, t, args=(alpha1, beta1, gamma1, d1))
x1, x2, x3, x4, x5 = sol1.T

plt.figure(figsize=(10, 6))
plt.plot(t, x1, label='金 (肺)', color='blue')
plt.plot(t, x2, label='水 (肾)', color='orange')
plt.plot(t, x3, label='木 (肝)', color='green')
plt.plot(t, x4, label='火 (心)', color='red')
plt.plot(t, x5, label='土 (脾)', color='purple')
plt.axhline(y=x_star1, color='gray', linestyle='--', linewidth=0.8,
            label=f'平衡点 x*={x_star1:.2f}')
plt.xlabel('时间 t')
plt.ylabel('功能态 x_i')
plt.title('图1 健康状态：稳定回归（肝亢20%扰动后系统自愈）')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.xlim(0, 50)
plt.tight_layout()
plt.savefig('images/figure1_healthy.png', dpi=300)
plt.close()
print("图1已保存至 images/figure1_healthy.png")


# ==================== 图2：病理状态肝阳上亢 ====================
print("生成图2...")
alpha2, beta2, gamma2, d2 = 0.4, 0.2, 0.25, 1.0
x_star2 = d2 / (gamma2 - alpha2 + beta2)

X0_2 = [x_star2, x_star2, x_star2 * 1.2, x_star2, x_star2]  # 肝+20%

sol2 = odeint(wuxing_model, X0_2, t, args=(alpha2, beta2, gamma2, d2))
x1, x2, x3, x4, x5 = sol2.T

plt.figure(figsize=(10, 6))
plt.plot(t, x1, label='金 (肺)', color='blue')
plt.plot(t, x2, label='水 (肾)', color='orange')
plt.plot(t, x3, label='木 (肝)', color='green')
plt.plot(t, x4, label='火 (心)', color='red')
plt.plot(t, x5, label='土 (脾)', color='purple')
plt.axhline(y=x_star2, color='gray', linestyle='--', linewidth=0.8,
            label=f'平衡点 x*={x_star2:.1f}')
plt.xlabel('时间 t')
plt.ylabel('功能态 x_i')
plt.title('图2 病理状态：肝阳上亢的模拟（参数不满足稳定判据）')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.xlim(0, 50)
plt.tight_layout()
plt.savefig('images/figure2_pathological.png', dpi=300)
plt.close()
print("图2已保存至 images/figure2_pathological.png")


# ==================== 图3：γ分岔图 ====================
print("生成图3（此计算可能需要几十秒）...")
alpha3, beta3, d3 = 0.2, 0.3, 1.0
gamma_th = 0.309 * alpha3 + 0.809 * beta3
print(f"理论阈值 γ_th = {gamma_th:.4f}")

# γ扫描范围（从大到小便于绘图）
gamma_values = np.linspace(0.6, 0.2, 200)
deviations = []

# 使用较稀疏的时间点以加快计算
t_short = np.linspace(0, 50, 200)

for gamma in gamma_values:
    x_star = d3 / (gamma - alpha3 + beta3)
    X0 = [x_star, x_star, x_star * 1.2, x_star, x_star]
    sol = odeint(wuxing_model, X0, t_short, args=(alpha3, beta3, gamma, d3))
    x3_final = sol[-1, 2]
    deviation = abs(x3_final - x_star)
    deviations.append(deviation)

deviations = np.array(deviations)

plt.figure(figsize=(10, 6))
plt.plot(gamma_values, deviations, 'b-', linewidth=2, label='肝的最终偏离')
plt.axvline(x=gamma_th, color='red', linestyle='--', linewidth=2,
            label=f'理论阈值 γ_th = {gamma_th:.3f}')
plt.xlabel('自稳力 γ')
plt.ylabel('t=50 时肝的绝对偏离 |x₃ - x*|')
plt.title('图3 参数变化与疾病传变：自稳力 γ 对系统稳定性的影响')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.xlim(0.2, 0.6)

# 添加稳定区/不稳定区标注
ymax = max(deviations)
plt.text(0.45, ymax*0.8, '稳定区', fontsize=12, ha='center',
         bbox=dict(facecolor='lightgreen', alpha=0.5))
plt.text(0.25, ymax*0.8, '不稳定区', fontsize=12, ha='center',
         bbox=dict(facecolor='lightcoral', alpha=0.5))

plt.tight_layout()
plt.savefig('images/figure3_bifurcation.png', dpi=300)
plt.close()
print("图3已保存至 images/figure3_bifurcation.png")
print("所有图形生成完毕！")
