import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import Dataset, DataLoader

def prepare_data(file_path, window_size=30):
    df = pd.read_csv(file_path, parse_dates=['date'])
    df.set_index('date', inplace=True)
    test_size = int(len(df) * 0.2)

    df['return'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['return'].rolling(window=20).std() * np.sqrt(240)
    df = df.dropna()

    def create_dataset(data, target_col, window_size):
        X, y = [], []
        for i in range(window_size, len(data) - 1):
            X.append(data[target_col].iloc[i - window_size:i].values)
            y.append(data[target_col].iloc[i + 1])
        return np.array(X), np.array(y)

    X_ret, y_ret = create_dataset(df, 'return', window_size)
    X_vol, y_vol = create_dataset(df, 'volatility', window_size)

    def split_data(data, test_size):
        return data[:-test_size], data[-test_size:]

    return (
        (*split_data(X_ret, test_size), *split_data(y_ret, test_size)),
        (*split_data(X_vol, test_size), *split_data(y_vol, test_size)),
        df.index[window_size + 1:-1],
        test_size
    )
class FunctionalBasis(nn.Module):
    def __init__(self, n_basis=10):
        super().__init__()
        self.basis_weights = nn.Parameter(torch.randn(n_basis))  # 可学习基函数

    def forward(self, x):
        # 使用傅里叶基函数展开
        t = torch.linspace(0, 1, x.shape[1]).to(x.device)
        basis = torch.stack([torch.sin(2 * np.pi * k * t) for k in range(self.basis_weights.shape[0])])
        return torch.einsum('bkt,k->bt', basis, self.basis_weights)  # 基展开

class DeepFunctionalFactorModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super().__init__()
        self.basis_expansion = FunctionalBasis()
        # 函数特征提取层：使用LSTM捕获时间序列的函数空间表示
        self.functional_layer = nn.LSTM(input_size, hidden_size, num_layers,
                                        batch_first=True, dropout=0.3 if num_layers > 1 else 0)

        # 注意力机制增强函数关系建模
        self.functional_attention = nn.MultiheadAttention(hidden_size, num_heads=4)

        # 深度非线性映射
        self.deep_mapping = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, output_size)
        )

    def forward(self, x):
        # 函数空间编码
        x, _ = self.functional_layer(x)  # [batch, seq_len, hidden_size]

        # 函数关系注意力
        x_attn, _ = self.functional_attention(x.transpose(0, 1), x.transpose(0, 1), x.transpose(0, 1))
        x = x + 0.5 * x_attn.transpose(0, 1)  # 残差连接

        return self.deep_mapping(x[:, -1, :])  # 从函数空间映射到预测目标

# 消融实验变体定义

class DeepFunctionalFactorModelV1(nn.Module):  # 变体1：移除注意力机制
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super().__init__()
        self.basis_expansion = FunctionalBasis()
        self.functional_layer = nn.LSTM(input_size, hidden_size, num_layers,
                                        batch_first=True, dropout=0.3 if num_layers > 1 else 0)
        self.deep_mapping = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, output_size))

    def forward(self, x):
        x, _ = self.functional_layer(x)
        return self.deep_mapping(x[:, -1, :])


class DeepFunctionalFactorModelV2(nn.Module):  # 变体2：简化深度映射层
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super().__init__()
        self.basis_expansion = FunctionalBasis()
        self.functional_layer = nn.LSTM(input_size, hidden_size, num_layers,
                                        batch_first=True, dropout=0.3 if num_layers > 1 else 0)
        self.functional_attention = nn.MultiheadAttention(hidden_size, num_heads=4)
        self.deep_mapping = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.functional_layer(x)
        x_attn, _ = self.functional_attention(x.transpose(0, 1), x.transpose(0, 1), x.transpose(0, 1))
        x = x + 0.5 * x_attn.transpose(0, 1)
        return self.deep_mapping(x[:, -1, :])


class DeepFunctionalFactorModelV3(nn.Module):  # 变体3：使用GRU代替LSTM
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super().__init__()
        self.basis_expansion = FunctionalBasis()
        self.functional_layer = nn.GRU(input_size, hidden_size, num_layers,
                                       batch_first=True, dropout=0.3 if num_layers > 1 else 0)
        self.functional_attention = nn.MultiheadAttention(hidden_size, num_heads=4)
        self.deep_mapping = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, output_size))

    def forward(self, x):
        x, _ = self.functional_layer(x)
        x_attn, _ = self.functional_attention(x.transpose(0, 1), x.transpose(0, 1), x.transpose(0, 1))
        x = x + 0.5 * x_attn.transpose(0, 1)
        return self.deep_mapping(x[:, -1, :])

def train_model(model, train_loader, device, epochs=100):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

    model.to(device)
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_model.pth')

        print(f'Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')
    return model
# 修改后的主函数
def main_ablation():
    WINDOW_SIZE = 30
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2

    # 数据形状 [samples, seq_len, features]
    def reshape_data(data):
        return data.reshape(-1, WINDOW_SIZE, 1)
    class MarketDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(reshape_data(X), dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)

        def __len__(self): return len(self.X)

        def __getitem__(self, idx): return self.X[idx], self.y[idx]
    # 准备数据（保持不变）
    (X_ret_train, X_ret_test, y_ret_train, y_ret_test), \
        (X_vol_train, X_vol_test, y_vol_train, y_vol_test), dates, test_size = prepare_data('data/SP500.csv',
                                                                                            WINDOW_SIZE)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # 定义消融实验配置
    experiments = [
        #('Original', DeepFunctionalFactorModel),
        ('NoAttention', DeepFunctionalFactorModelV1),
        ('SimpleMapping', DeepFunctionalFactorModelV2),
        ('GRU', DeepFunctionalFactorModelV3)
    ]
    def predict(model, X_test):
        with torch.no_grad():
            test_tensor = torch.tensor(reshape_data(X_test),
                                       dtype=torch.float32).to(device)
            return model(test_tensor).cpu().numpy().flatten()
    # 运行所有实验
    for model_name, model_class in experiments:
        print(f"\n=== Running Experiment: {model_name} ===")

        # 收益率预测
        ret_train_dataset = MarketDataset(X_ret_train[:-1], y_ret_train[1:])
        ret_train_loader = DataLoader(ret_train_dataset, batch_size=64, shuffle=True, drop_last=True)
        model_ret = model_class(input_size=1, hidden_size=HIDDEN_SIZE,
                                output_size=1, num_layers=NUM_LAYERS)
        train_model(model_ret, ret_train_loader, device)

        # 波动率预测
        vol_train_dataset = MarketDataset(X_vol_train[:-1], y_vol_train[1:])
        vol_train_loader = DataLoader(vol_train_dataset, batch_size=64, shuffle=True, drop_last=True)
        model_vol = model_class(input_size=1, hidden_size=HIDDEN_SIZE,
                                output_size=1, num_layers=NUM_LAYERS)
        train_model(model_vol, vol_train_loader, device)

        # 预测
        pred_ret = predict(model_ret, X_ret_test)
        pred_vol = predict(model_vol, X_vol_test)

        # 评估指标
        print(f"\n{model_name} Model Performance:")

        def print_metrics(name, true, pred):
            print(f'{name} Metrics:')
            print(f'MSE: {mean_squared_error(true, pred):.4f}')
            print(f'RMSE: {np.sqrt(mean_squared_error(true, pred)):.4f}')
            print(f'MAE: {mean_absolute_error(true, pred):.4f}')
            print(f'R²: {r2_score(true, pred):.4f}\n')
        print_metrics('Return', y_ret_test, pred_ret)
        print_metrics('Volatility', y_vol_test, pred_vol)

        # 可视化（可选）
        # plt.figure(figsize=(12, 6))
        # plt.plot(y_ret_test, label='Actual')
        # plt.plot(pred_ret, label='Predicted', alpha=0.7)
        # plt.title(f'{model_name} Return Predictions')
        # plt.legend()
        # plt.show()


if __name__ == '__main__':
    main_ablation()  # 运行消融实验主函数