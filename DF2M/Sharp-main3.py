# 深度函数因子模型 DeepFunctionalFactorModel
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import Dataset, DataLoader


# 数据预处理
def prepare_data(file_path, window_size=30):
    df = pd.read_csv(file_path, parse_dates=['date'])
    df.set_index('date', inplace=True)

    df['return'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['return'].rolling(window=20).std() * np.sqrt(252)
    df['sharpe'] = df['return'] / df['volatility']  # 计算夏普比率
    df.replace([np.inf, -np.inf], np.nan, inplace=True)  # 处理无穷值
    df = df.dropna()

    def create_dataset(data, target_col, window_size):
        X, y = [], []
        for i in range(window_size, len(data) - 1):
            X.append(data[target_col].iloc[i - window_size:i].values)
            y.append(data[target_col].iloc[i + 1])
        return np.array(X), np.array(y)

    X_ret, y_ret = create_dataset(df, 'return', window_size)
    X_vol, y_vol = create_dataset(df, 'volatility', window_size)
    X_sharpe, y_sharpe = create_dataset(df, 'sharpe', window_size)  # 夏普数据集

    def split_data(data, test_size_ratio=0.2):
        test_size = int(len(data) * test_size_ratio)
        return data[:-test_size], data[-test_size:]

    return (
        (*split_data(X_ret), *split_data(y_ret)),
        (*split_data(X_vol), *split_data(y_vol)),
        (*split_data(X_sharpe), *split_data(y_sharpe)),  # 夏普数据拆分
        df.index[window_size + 1:-1],
    )


# 函数基展开层（保持不变）
class FunctionalBasis(nn.Module):
    def __init__(self, n_basis=10):
        super().__init__()
        self.basis_weights = nn.Parameter(torch.randn(n_basis))

    def forward(self, x):
        t = torch.linspace(0, 1, x.shape[1]).to(x.device)
        basis = torch.stack([torch.sin(2 * np.pi * k * t) for k in range(self.basis_weights.shape[0])])
        return torch.einsum('bkt,k->bt', basis, self.basis_weights)


class DeepFunctionalFactorModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super().__init__()
        self.basis_expansion = FunctionalBasis()
        self.functional_layer = nn.LSTM(input_size, hidden_size, num_layers,
                                        batch_first=True, dropout=0.3 if num_layers > 1 else 0)
        self.functional_attention = nn.MultiheadAttention(hidden_size, num_heads=4)
        self.deep_mapping = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, output_size)
        )

    def forward(self, x):
        x, _ = self.functional_layer(x)
        x_attn, _ = self.functional_attention(x.transpose(0, 1), x.transpose(0, 1), x.transpose(0, 1))
        x = x + 0.5 * x_attn.transpose(0, 1)
        return self.deep_mapping(x[:, -1, :])


# 训练函数（保持不变）
def train_model(model, train_loader, device, epochs=100):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
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

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_model.pth')

        print(f'Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')
    return model


# 主流程
def main():
    WINDOW_SIZE = 30
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2

    # 准备数据
    ((X_ret_train, X_ret_test, y_ret_train, y_ret_test),
     (X_vol_train, X_vol_test, y_vol_train, y_vol_test),
     (X_sharpe_train, X_sharpe_test, y_sharpe_train, y_sharpe_test),
     dates) = prepare_data('data/NASDAQ_100.csv', WINDOW_SIZE)

    # 设备配置
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # 数据形状处理
    def reshape_data(data):
        return data.reshape(-1, WINDOW_SIZE, 1)

    class MarketDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(reshape_data(X), dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)

        def __len__(self): return len(self.X)

        def __getitem__(self, idx): return self.X[idx], self.y[idx]

    # 训练三个模型
    def train_and_predict(X_train, y_train, X_test, model_name):
        dataset = MarketDataset(X_train[:-1], y_train[1:])
        loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)
        model = DeepFunctionalFactorModel(1, HIDDEN_SIZE, 1, NUM_LAYERS)
        model = train_model(model, loader, device)
        return predict(model, X_test)

    def predict(model, X_test):
        with torch.no_grad():
            test_tensor = torch.tensor(reshape_data(X_test), dtype=torch.float32).to(device)
            return model(test_tensor).cpu().numpy().flatten()

    # 训练并预测
    pred_ret = train_and_predict(X_ret_train, y_ret_train, X_ret_test, "Return")
    pred_vol = train_and_predict(X_vol_train, y_vol_train, X_vol_test, "Volatility")
    pred_sharpe = train_and_predict(X_sharpe_train, y_sharpe_train, X_sharpe_test, "Sharpe Ratio")

    # 可视化
    test_size = len(X_ret_test)
    test_dates = dates[-test_size:]

    plt.figure(figsize=(15, 12))
    plt.suptitle('NASDAQ100 Predictions in DFFM', fontsize=16)

    metrics = []
    for i, (name, true, pred) in enumerate([
        ('Return', y_ret_test, pred_ret),
        ('Volatility', y_vol_test, pred_vol),
        ('Sharpe Ratio', y_sharpe_test, pred_sharpe)
    ]):
        plt.subplot(3, 1, i + 1)
        plt.plot(test_dates, true, label=f'Actual {name}')
        plt.plot(test_dates, pred, label=f'Predicted {name}', alpha=0.7)
        plt.title(f'{name} Predictions')
        plt.legend()
        metrics.append((name, true, pred))

    plt.tight_layout()
    plt.savefig('results/df2m/dffm_predictions_with_sharpe.png')
    plt.show()

    # 评估指标
    def print_metrics(name, true, pred):
        print(f'\n{name} Metrics:')
        print(f'MSE: {mean_squared_error(true, pred):.4f}')
        print(f'RMSE: {np.sqrt(mean_squared_error(true, pred)):.4f}')
        print(f'MAE: {mean_absolute_error(true, pred):.4f}')
        print(f'R²: {r2_score(true, pred):.4f}')

    for name, true, pred in metrics:
        print_metrics(name, true, pred)


if __name__ == '__main__':
    main()