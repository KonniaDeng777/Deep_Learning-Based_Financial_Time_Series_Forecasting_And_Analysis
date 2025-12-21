import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, SimpleRNN, GRU, BatchNormalization, \
    Bidirectional, Flatten
from keras.callbacks import EarlyStopping

def prepare_data(file_path, window_size=30, future_days=5):
    # 读取数据
    df = pd.read_csv(file_path, parse_dates=['date'])
    df.sort_values('date', inplace=True)
    df.set_index('date', inplace=True)

    # 计算目标变量
    df['return_target'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility_target'] = df['return_target'].rolling(window=20).std() * np.sqrt(240)

    # 删除无效数据
    df.dropna(subset=['return_target', 'volatility_target'], inplace=True)

    # 特征选择
    features = ['open', 'high', 'low', 'close', 'volume', 'change_percent', 'avg_vol_20d']
    X = df[features]
    y = df[['return_target', 'volatility_target']]

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 创建时间窗口
    X_data, y_data, dates = [], [], []
    for i in range(window_size, len(X_scaled)):
        X_data.append(X_scaled[i - window_size:i])
        y_data.append(y.iloc[i].values)
        dates.append(df.index[i])  # Store the date corresponding to the current window

    return np.array(X_data), np.array(y_data), np.array(dates)

# 构建模型
# def build_model(input_shape):
#     inputs = Input(shape=input_shape)
#
#     # CNN部分
#     x = Conv1D(128, 3, activation='tanh', padding='same')(inputs)
#     # x = BatchNormalization()(x)
#     x = MaxPooling1D(2)(x)
#     x = Dropout(0.3)(x)
#
#     # 增加一层卷积
#     x = Conv1D(128, 3, activation='tanh', padding='same')(x)
#     # x = BatchNormalization()(x)
#     x = MaxPooling1D(2)(x)
#     x = Dropout(0.3)(x)
#
#     # RNN部分
#     x = SimpleRNN(100, return_sequences=True)(x)
#     x = SimpleRNN(50)(x)
#     x = Dropout(0.3)(x)
#
#     # # RNN部分（可以使用GRU或Bidirectional RNN）
#     # x = Bidirectional(GRU(100, return_sequences=True))(x)  # 使用双向GRU
#     # x = GRU(50)(x)  # 使用GRU替代SimpleRNN
#     # x = Dropout(0.3)(x)
#
#     # 输出层
#     output_return = Dense(1, name='return')(x)
#     output_volatility = Dense(1, name='volatility')(x)
#
#     model = Model(inputs=inputs, outputs=[output_return, output_volatility])
#
#     model.compile(optimizer='rmsprop',
#                   loss={'return': 'mse', 'volatility': 'mse'},
#                   loss_weights=[0.5, 0.5],
#                   metrics={'return': ['mae'], 'volatility': ['mae']})
#
#     return model

def build_model(input_shape):
    inputs = Input(shape=input_shape)

    # CNN 部分 (更深的 CNN 结构)
    x = Conv1D(64, kernel_size=5, activation='relu', padding='same')(inputs)
    # x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.1)(x)

    x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    # x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.1)(x)

    x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    # x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.1)(x)

    # RNN 部分 (优化 SimpleRNN)
    x = SimpleRNN(64, return_sequences=True)(x)
    x = SimpleRNN(32)(x)
    x = Dropout(0.1)(x)

    # 全局特征提取
    x = Flatten()(x)

    # 输出层
    output_return = Dense(1, name='return')(x)
    output_volatility = Dense(1, name='volatility')(x)

    model = Model(inputs=inputs, outputs=[output_return, output_volatility])

    model.compile(optimizer='adam',
                  loss={'return': 'mse', 'volatility': 'mse'},
                  loss_weights=[0.5, 0.5],
                  metrics={'return': ['mae'], 'volatility': ['mae']})

    return model


def calculate_information_ratio(actual_returns, predicted_returns):
    """计算 Information Ratio (IR)"""
    mean_excess_return = np.mean(predicted_returns - actual_returns)
    tracking_error = np.std(predicted_returns - actual_returns)

    if tracking_error == 0:
        return np.nan  # 避免除以零

    return mean_excess_return / tracking_error

# 主程序
def main():
    # 参数设置
    WINDOW_SIZE = 30
    FUTURE_DAYS = 5
    TEST_SIZE = 0.2

    # 准备数据
    X, y,dates = prepare_data('data/SP500.csv', WINDOW_SIZE, FUTURE_DAYS)

    # 划分数据集
    split = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    y_return_train, y_volatility_train = y_train[:, 0], y_train[:, 1]
    y_return_test, y_volatility_test = y_test[:, 0], y_test[:, 1]
    dates_test = dates[split:]
    # 构建模型
    model = build_model((WINDOW_SIZE, X.shape[2]))

    # 训练模型
    history = model.fit(X_train,
                        {'return': y_return_train, 'volatility': y_volatility_train},
                        epochs=500,
                        batch_size=64,
                        validation_split=0.1,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=3)],
                        verbose=1)

    # 预测结果
    predictions = model.predict(X_test)
    pred_return = predictions[0].flatten()
    pred_volatility = predictions[1].flatten()

    # 计算性能指标
    print("\nReturn Prediction Metrics:")
    print(f"MAE: {mean_absolute_error(y_return_test, pred_return):.4f}")
    print(f"MSE: {mean_squared_error(y_return_test, pred_return):.4f}")
    print(f"RMSE: {mean_squared_error(y_return_test, pred_return) ** 0.5:.4f}")
    print(f"R²: {r2_score(y_return_test, pred_return):.4f}")
    print(f"Information Ratio: {calculate_information_ratio(y_return_test, pred_return):.4f}")

    print("\nVolatility Prediction Metrics:")
    print(f"MAE: {mean_absolute_error(y_volatility_test, pred_volatility):.4f}")
    print(f"MSE: {mean_squared_error(y_volatility_test, pred_volatility):.4f}")
    print(f"RMSE: {mean_squared_error(y_volatility_test, pred_volatility) ** 0.5:.4f}")
    print(f"R²: {r2_score(y_volatility_test, pred_volatility):.4f}")
    print(f"Information Ratio: {calculate_information_ratio(y_volatility_test, pred_volatility):.4f}")

    # 绘制结果
    plt.figure(figsize=(15, 6))
    plt.suptitle('S&P500 Predictions in CNN-RNN Model', fontsize=16)
    plt.subplot(2, 1, 1)
    plt.plot(dates_test,y_return_test, label='Actual Returns',color='#4A4A90')
    plt.plot(dates_test,pred_return, label='Predicted Returns',color='#E8AC2F')
    plt.title('Return Predictions')
    plt.xlabel('Time')
    plt.ylabel('Return')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(dates_test,y_volatility_test, label='Actual Volatility',color='#4A4A90')
    plt.plot(dates_test,pred_volatility, label='Predicted Volatility',color='#E8AC2F')
    plt.title('Volatility Predictions')
    plt.xlabel('Time')
    plt.ylabel('Volatility')
    plt.legend()

    plt.tight_layout()
    plt.savefig('results/SP-predictions-RNN.png')
    plt.show()


if __name__ == "__main__":
    main()