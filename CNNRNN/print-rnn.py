import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 数据表
metrics = ['MAE', 'MSE', 'RMSE', 'R²']
categories = ['S&P500 (Return)', 'SSE Index (Return)', 'S&P500 (Volatility)', 'SSE Index (Volatility)']
models = ['CNN-RNN', 'CNN', 'RNN']

data = {
    'S&P500 (Return)': [
        [0.0104, 0.0002, 0.0146, -0.7659],  # CNN-RNN
        [0.0122, 0.0002, 0.0153, -0.9462],  # CNN Only
        [0.0980, 0.0142, 0.1192, -116.3886]  # RNN Only
    ],
    'SSE Index (Return)': [
        [0.0115, 0.0003, 0.0158, -0.4800],  # CNN-RNN
        [0.0105, 0.0002, 0.0144, -0.2231],  # CNN Only
        [0.0262, 0.0020, 0.0442, -10.5427]  # RNN Only
    ],
    'S&P500 (Volatility)': [
        [0.0631, 0.0054, 0.0735, 0.3785],  # CNN-RNN
        [0.0706, 0.0086, 0.0929, -0.0427],  # CNN Only
        [0.2033, 0.0573, 0.2394, -5.9312]  # RNN Only
    ],
    'SSE Index (Volatility)': [
        [0.0251, 0.0014, 0.0370, 0.8238],  # CNN-RNN
        [0.0285, 0.0017, 0.0415, 0.7792],  # CNN Only
        [0.2063, 0.0500, 0.2235, -5.4152]  # RNN Only
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
plt.savefig('results/compare-rnn.png')
plt.show()