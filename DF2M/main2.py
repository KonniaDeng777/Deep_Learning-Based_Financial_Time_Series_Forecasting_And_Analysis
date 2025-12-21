# 深度函数因子模型 Deep Factor Model
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
    test_size = int(len(df) * 0.2)

    df['return'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['return'].rolling(window=20).std() * np.sqrt(252)
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


# 深度函数因子模型
class DeepFactorModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.3 if num_layers > 1 else 0)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, output_size)
        )

    def forward(self, x):
        x, _ = self.lstm(x)  # [batch, seq_len, hidden_size]

        # 注意力机制
        x_attn, _ = self.attention(x.transpose(0, 1), x.transpose(0, 1), x.transpose(0, 1))
        x = x + 0.5 * x_attn.transpose(0, 1)

        return self.fc(x[:, -1, :])  # 取最后一个时间步


# 训练函数
def train_model(model, train_loader, device, epochs=150):
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

        # Early stopping
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
    (X_ret_train, X_ret_test, y_ret_train, y_ret_test), \
        (X_vol_train, X_vol_test, y_vol_train, y_vol_test), dates, test_size = prepare_data('data/SP500.csv', WINDOW_SIZE)

    # 设备配置
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # 数据形状 [samples, seq_len, features]
    def reshape_data(data):
        return data.reshape(-1, WINDOW_SIZE, 1)

    class MarketDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(reshape_data(X), dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)

        def __len__(self): return len(self.X)

        def __getitem__(self, idx): return self.X[idx], self.y[idx]

    # 收益率预测
    ret_train_dataset = MarketDataset(X_ret_train[:-1], y_ret_train[1:])
    ret_train_loader = DataLoader(ret_train_dataset, batch_size=64, shuffle=True, drop_last=True)

    model_ret = DeepFactorModel(input_size=1, hidden_size=HIDDEN_SIZE,
                                output_size=1, num_layers=NUM_LAYERS)
    model_ret = train_model(model_ret, ret_train_loader, device)

    # 波动率预测
    vol_train_dataset = MarketDataset(X_vol_train[:-1], y_vol_train[1:])
    vol_train_loader = DataLoader(vol_train_dataset, batch_size=64, shuffle=True, drop_last=True)

    model_vol = DeepFactorModel(input_size=1, hidden_size=HIDDEN_SIZE,
                                output_size=1, num_layers=NUM_LAYERS)
    model_vol = train_model(model_vol, vol_train_loader, device)

    # 预测
    def predict(model, X_test):
        with torch.no_grad():
            test_tensor = torch.tensor(reshape_data(X_test),
                                       dtype=torch.float32).to(device)
            return model(test_tensor).cpu().numpy().flatten()

    pred_ret = predict(model_ret, X_ret_test)
    pred_vol = predict(model_vol, X_vol_test)

    # 可视化
    test_dates = dates[-test_size:]

    plt.figure(figsize=(15, 8))
    plt.suptitle('S&P500 Predictions in Deep Factor Model', fontsize=16)

    plt.subplot(2, 1, 1)
    plt.plot(test_dates, y_ret_test, label='Actual Returns')
    plt.plot(test_dates, pred_ret, label='Predicted Returns', alpha=0.7)
    plt.title('Return Predictions')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(test_dates, y_vol_test, label='Actual Volatility')
    plt.plot(test_dates, pred_vol, label='Predicted Volatility', alpha=0.7)
    plt.title('Volatility Predictions')
    plt.legend()

    plt.tight_layout()
    plt.savefig('results/dfm/predictions.png')
    plt.show()

    def print_metrics(name, true, pred):
        print(f'{name} Metrics:')
        print(f'MSE: {mean_squared_error(true, pred):.4f}')
        print(f'RMSE: {np.sqrt(mean_squared_error(true, pred)):.4f}')
        print(f'MAE: {mean_absolute_error(true, pred):.4f}')
        print(f'R²: {r2_score(true, pred):.4f}\n')

    print_metrics('Return', y_ret_test, pred_ret)
    print_metrics('Volatility', y_vol_test, pred_vol)


if __name__ == '__main__':
    main()