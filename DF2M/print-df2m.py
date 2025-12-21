import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 数据表
metrics = ['MAE', 'MSE', 'RMSE', 'R²']
categories = ['S&P500 (Return)', 'SSE Index (Return)', 'S&P500 (Volatility)', 'SSE Index (Volatility)']
models = ['DF²M', 'No-Attention', 'Simple-Mapping']

data = {
    'S&P500 (Return)': [
        [0.0073, 0.0001, 0.0110, -0.0002],  # DF²M
        [0.0093, 0.0002, 0.0128, -0.3544],  # No-Attention
        [0.0091, 0.0002, 0.0126, -0.3049]   # Simple-Mapping
    ],
    'SSE Index (Return)': [
        [0.0090, 0.0002, 0.0130, -0.0020],   # DF²M
        [0.0131, 0.0003, 0.0173, -0.7728],  # No-Attention
        [0.0107, 0.0002, 0.0145, -0.2562]   # Simple-Mapping
    ],
    'S&P500 (Volatility)': [
        [0.0175, 0.0007, 0.0270, 0.9157],   # DF²M
        [0.0381, 0.0034, 0.0586, 0.5845],   # No-Attention
        [0.0316, 0.0027, 0.0517, 0.6767]    # Simple-Mapping
    ],
    'SSE Index (Volatility)': [
        [0.0266, 0.0014, 0.0378, 0.8147],   # DF²M
        [0.0577, 0.0079, 0.0890, -0.0275],  # No-Attention
        [0.1012, 0.0124, 0.1112, -0.6047]   # Simple-Mapping
    ]
}

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
colors = ['#487DB2', '#7698C3', '#AABCDB']
x = np.arange(len(categories))
width = 0.25

# 遍历每个指标进行绘图
for i, metric in enumerate(metrics):
    ax = axes[i]
    for j, model in enumerate(models):
        values = [data[cat][j][i] for cat in categories]
        ax.bar(x + j * width, values, width, label=model, color=colors[j])

    ax.set_xticks(x + width)
    ax.set_xticklabels(categories, rotation=20, ha='right')
    ax.set_title(metric)
    ax.legend()

plt.tight_layout()
plt.savefig('results/df2m/compare-df2m.png')
plt.show()