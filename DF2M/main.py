# 因子分解预测模型Factor Model
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import Dataset, DataLoader


# 数据预处理
def prepare_data(file_path, window_size=30):
    # 读取数据
    df = pd.read_csv(file_path, parse_dates=['date'])
    df.set_index('date', inplace=True)
    test_size=int(len(df) * 0.2)
    # 计算波动率（5日滚动标准差）
    #df['volatility'] = df['change_percent'].rolling(5).std()
    df['return'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['return'].rolling(window=20).std() * np.sqrt(252)
    df = df.dropna()

    # 创建数据集
    def create_dataset(data, target_col, window_size):
        X, y = [], []
        for i in range(window_size, len(data) - 1):
            X.append(data[target_col].iloc[i - window_size:i].values)
            y.append(data[target_col].iloc[i + 1])
        return np.array(X), np.array(y)

    # 收益率数据
    X_ret, y_ret = create_dataset(df, 'return', window_size)
    # 波动率数据
    X_vol, y_vol = create_dataset(df, 'volatility', window_size)

    # 划分训练测试集
    def split_data(data, test_size):
        return data[:-test_size], data[-test_size:]

    return (
        (*split_data(X_ret, test_size), *split_data(y_ret, test_size)),
        (*split_data(X_vol, test_size), *split_data(y_vol, test_size)),
        df.index[window_size + 1:-1],
        test_size
    )


# 因子分解预测模型
class FactorForecastModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])


# 训练函数
def train_model(model, train_loader, device, epochs=100):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.to(device)

    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}')
    return model


# 主流程
def main():
    # 配置参数
    WINDOW_SIZE = 30
    TEST_SIZE = 100
    N_COMPONENTS = 5
    HIDDEN_SIZE = 32

    # 准备数据
    (X_ret_train, X_ret_test, y_ret_train, y_ret_test),(X_vol_train, X_vol_test, y_vol_train, y_vol_test),dates,TEST_SIZE = prepare_data('data/SP500.csv', WINDOW_SIZE)

    # 设备配置
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # 收益率预测
    # PCA分解
    pca_ret = PCA(N_COMPONENTS)
    X_ret_train_pca = pca_ret.fit_transform(X_ret_train.reshape(-1, WINDOW_SIZE))
    X_ret_test_pca = pca_ret.transform(X_ret_test.reshape(-1, WINDOW_SIZE))

    # 创建数据集
    class FactorDataset(Dataset):
        def __init__(self, X_pca, y):
            self.X = torch.tensor(X_pca, dtype=torch.float32).unsqueeze(1)
            self.y = torch.tensor(y, dtype=torch.float32)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    # 训练模型
    ret_train_dataset = FactorDataset(X_ret_train_pca[:-1], y_ret_train[1:])
    ret_train_loader = DataLoader(ret_train_dataset, batch_size=32, shuffle=True)
    model_ret = FactorForecastModel(N_COMPONENTS, HIDDEN_SIZE, 1)
    model_ret = train_model(model_ret, ret_train_loader, device)

    # 测试预测
    with torch.no_grad():
        test_inputs = torch.tensor(X_ret_test_pca, dtype=torch.float32).unsqueeze(1).to(device)
        pred_ret = model_ret(test_inputs).cpu().numpy().flatten()

    # 波动率预测（类似流程）
    pca_vol = PCA(N_COMPONENTS)
    X_vol_train_pca = pca_vol.fit_transform(X_vol_train.reshape(-1, WINDOW_SIZE))
    X_vol_test_pca = pca_vol.transform(X_vol_test.reshape(-1, WINDOW_SIZE))

    vol_train_dataset = FactorDataset(X_vol_train_pca[:-1], y_vol_train[1:])
    vol_train_loader = DataLoader(vol_train_dataset, batch_size=32, shuffle=True)
    model_vol = FactorForecastModel(N_COMPONENTS, HIDDEN_SIZE, 1)
    model_vol = train_model(model_vol, vol_train_loader, device)

    with torch.no_grad():
        test_inputs = torch.tensor(X_vol_test_pca, dtype=torch.float32).unsqueeze(1).to(device)
        pred_vol = model_vol(test_inputs).cpu().numpy().flatten()

    # 可视化结果
    test_dates = dates[-TEST_SIZE:]

    plt.figure(figsize=(15, 6))


    plt.subplot(2, 1, 1)
    plt.suptitle('S&P500 Predictions in Factor Model',fontsize=16)
    plt.plot(test_dates, y_ret_test, label='Actual Returns')
    plt.plot(test_dates, pred_ret, label='Predicted Returns')
    plt.title('Return Predictions')
    plt.xlabel('Time')
    plt.ylabel('Return')
    plt.legend()


    plt.subplot(2, 1, 2)
    plt.plot(test_dates, y_vol_test, label='Actual Volatility')
    plt.plot(test_dates, pred_vol, label='Predicted Volatility')
    plt.title('Volatility Predictions')
    plt.xlabel('Time')
    plt.ylabel('Volatility')
    plt.legend()

    plt.tight_layout()
    plt.savefig('results/fm/predictions-FM.png')
    plt.show()

    # 计算指标
    def print_metrics(name, true, pred):
        rmse = np.sqrt(mean_squared_error(true, pred))
        mae = mean_absolute_error(true, pred)
        mse = mean_squared_error(true, pred)
        r_score = r2_score(true, pred)
        print(f'{name} Metrics:')
        print(f'MSE: {mse:.4f}')
        print(f'RMSE: {rmse:.4f}')
        print(f'MAE: {mae:.4f}')
        print(f'R²: {r_score:.4f}\n')

    print_metrics('Return', y_ret_test, pred_ret)
    print_metrics('Volatility', y_vol_test, pred_vol)


if __name__ == '__main__':
    main()