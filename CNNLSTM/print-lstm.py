import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 数据表
metrics = ['MAE', 'MSE', 'RMSE', 'R²']
categories = ['S&P500 (Return)', 'SSE Index (Return)', 'S&P500 (Volatility)', 'SSE Index (Volatility)']
models = ['CNN-LSTM', 'CNN', 'LSTM']

data = {
    'S&P500 (Return)': [
        [0.0074, 0.0001, 0.0113, -0.0587],  # CNN-LSTM
        [0.0122, 0.0002, 0.0153, -0.9462],  # CNN Only
        [0.0179, 0.0005, 0.0232, -3.4485]  # LSTM Only
    ],
    'SSE Index (Return)': [
        [0.0095, 0.0002, 0.0133, -0.0522],
        [0.0105, 0.0002, 0.0144, -0.2231],
        [0.0121, 0.0003, 0.0163, -0.5658]
    ],
    'S&P500 (Volatility)': [
        [0.0403, 0.0022, 0.0471, 0.7443],
        [0.0706, 0.0086, 0.0929, -0.0427],
        [0.0521, 0.0041, 0.0640, 0.5038]
    ],
    'SSE Index (Volatility)': [
        [0.0423, 0.0027, 0.0522, 0.6503],
        [0.0285, 0.0017, 0.0415, 0.7792],
        [0.0274, 0.0017, 0.0411, 0.7831]
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
plt.savefig('results/compare-lstm.png')
plt.show()