import keras.initializers.initializers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Concatenate, Flatten, Reshape, LeakyReLU, Dropout
from keras.optimizers import Adam, RMSprop

def prepare_data(file_path, window_size=30, forecast_horizon=5):
    df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')

    # 计算收益率和波动率
    df['return'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['return'].rolling(window=20).std() * np.sqrt(240)
    df.dropna(inplace=True)

    features = ['return', 'volatility', 'volume']
    #features = ['open', 'high', 'low', 'close', 'volume', 'change_percent', 'avg_vol_20d']
    targets = ['return', 'volatility']

    def create_dataset(data, target):
        X, y = [], []
        for i in range(len(data) - window_size - forecast_horizon + 1):
            X.append(data[i:i + window_size])
            y.append(target[i + window_size:i + window_size + forecast_horizon])
        return np.array(X), np.array(y)

    X, y = create_dataset(df[features].values, df[targets].values)
    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler_X.fit_transform(X.reshape(-1, X.shape[2])).reshape(X.shape)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, y.shape[2])).reshape(y.shape)

    return X_scaled, y_scaled, scaler_X, scaler_y, df.index[-len(y):]

# 模型构建
class FinGAN:
    def __init__(self, window_size, feature_dim, target_dim, forecast_horizon):
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.target_dim = target_dim
        self.forecast_horizon = forecast_horizon

        # 构建模型组件
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.combined = self.build_combined()

    def build_generator(self):
        initializer = keras.initializers.initializers.GlorotNormal()

        input_layer = Input(shape=(self.window_size, self.feature_dim))
        x = LSTM(256, return_sequences=True, kernel_initializer=initializer)(input_layer)
        x = LSTM(128, kernel_initializer=initializer)(x)
        x = Dense(256, activation='relu', kernel_initializer=initializer)(x)
        output = Dense(self.forecast_horizon * self.target_dim, activation='tanh', kernel_initializer=initializer)(x)
        output = Reshape((self.forecast_horizon, self.target_dim))(output)
        return Model(input_layer, output)

    def build_discriminator(self):
        initializer = keras.initializers.initializers.GlorotNormal()

        hist_input = Input(shape=(self.window_size, self.feature_dim))
        target_input = Input(shape=(self.forecast_horizon, self.target_dim))

        # 历史数据处理
        x = LSTM(128, kernel_initializer=initializer)(hist_input)

        # 目标数据处理
        y = Flatten()(target_input)

        combined = Concatenate()([x, y])
        x = Dense(256, kernel_initializer=initializer)(combined)
        x = LeakyReLU(0.2)(x)
        x = Dropout(0.1)(x)
        x = Dense(128, kernel_initializer=initializer)(x)
        x = LeakyReLU(0.2)(x)
        x = Dropout(0.1)(x)
        validity = Dense(1, activation='sigmoid', kernel_initializer=initializer)(x)

        return Model([hist_input, target_input], validity)

    def build_combined(self):
        self.discriminator.trainable = False
        hist_input = Input(shape=(self.window_size, self.feature_dim))
        generated_target = self.generator(hist_input)
        validity = self.discriminator([hist_input, generated_target])
        return Model(hist_input, [validity, generated_target])

    def train(self, X_train, y_train, epochs=200, batch_size=32):
        # 编译判别器 lr=1e-4
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=RMSprop(learning_rate=1e-4, rho=0.9),
                                   metrics=['accuracy'])

        # 编译组合模型 lr=1e-4
        self.combined.compile(loss=['binary_crossentropy', 'mse'],
                              loss_weights=[1, 100],
                              optimizer=RMSprop(learning_rate=1e-4, rho=0.9))

        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # 训练判别器
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_seqs = X_train[idx]
            real_targets = y_train[idx]

            fake_targets = self.generator.predict(real_seqs)

            d_loss_real = self.discriminator.train_on_batch([real_seqs, real_targets], real)
            d_loss_fake = self.discriminator.train_on_batch([real_seqs, fake_targets], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # 训练生成器
            g_loss = self.combined.train_on_batch(real_seqs, [real, real_targets])

            print(
                f"Epoch {epoch + 1}/{epochs} [D loss: {d_loss[0]:.4f} acc: {100 * d_loss[1]:.2f}%] [G loss: {g_loss[0]:.4f} mse: {g_loss[2]:.4f}]")


def calculate_information_ratio(actual_returns, predicted_returns):
    """计算 Information Ratio (IR)"""
    mean_excess_return = np.mean(predicted_returns - actual_returns)
    tracking_error = np.std(predicted_returns - actual_returns)

    if tracking_error == 0:
        return np.nan  # 避免除以零

    return mean_excess_return / tracking_error

if __name__ == "__main__":
    WINDOW_SIZE = 30
    FORECAST_HORIZON = 5
    EPOCHS = 1600
    BATCH_SIZE = 128

    X, y, scaler_X, scaler_y, timestamps = prepare_data('data/CSI500.csv', window_size=WINDOW_SIZE,
                                                        forecast_horizon=FORECAST_HORIZON)

    fgan = FinGAN(WINDOW_SIZE, X.shape[2], y.shape[2], FORECAST_HORIZON)
    fgan.train(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE)

    y_pred_scaled = fgan.generator.predict(X)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, y.shape[2])).reshape(y_pred_scaled.shape)
    y_real = scaler_y.inverse_transform(y.reshape(-1, y.shape[2])).reshape(y.shape)

    test_size = int(len(y) * 0.2)
    y_real_test = y_real[-test_size:]
    y_pred_test = y_pred[-test_size:]
    timestamps_test = timestamps[-test_size:]

    # # 计算评估指标
    # mae = mean_absolute_error(y_real_test.flatten(), y_pred_test.flatten())
    # mse = mean_squared_error(y_real_test.flatten(), y_pred_test.flatten())
    # rmse = np.sqrt(mse)
    # r2 = r2_score(y_real_test.flatten(), y_pred_test.flatten())
    #
    # # 输出结果
    # print(f"MAE: {mae:.4f}")
    # print(f"MSE: {mse:.4f}")
    # print(f"RMSE: {rmse:.4f}")
    # print(f"R²: {r2:.4f}")

    # 拆分 return 和 volatility
    return_real, volatility_real = y_real_test[:, :, 0], y_real_test[:, :, 1]
    return_pred, volatility_pred = y_pred_test[:, :, 0], y_pred_test[:, :, 1]

    def evaluate(y_true, y_pred, name):
        mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
        mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true.flatten(), y_pred.flatten())
        ir = calculate_information_ratio(y_true.flatten(), y_pred.flatten())
        print(f"{name} Information Ratio:{ir:.4f}")
        print(f"{name} MAE: {mae:.4f}")
        print(f"{name} MSE: {mse:.4f}")
        print(f"{name} RMSE: {rmse:.4f}")
        print(f"{name} R²: {r2:.4f}")

    # 分别计算 return 和 volatility 的评估指标
    evaluate(return_real, return_pred, "Return")
    evaluate(volatility_real, volatility_pred, "Volatility")

    plt.figure(figsize=(15, 6))
    plt.suptitle('SSE Index Predictions in GANs Model', fontsize=16)  # 主标题

    plt.subplot(2, 1, 1)
    plt.plot(timestamps_test, y_real_test[:, 0, 0].flatten(),  label='Actual Returns',color='#4A4A90')
    plt.plot(timestamps_test, y_pred_test[:, 0, 0].flatten(),  label='Predicted Returns',color='#E8AC2F')
    plt.title('Return Predictions')
    plt.xlabel('Time')
    plt.ylabel('Return')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(timestamps_test, y_real_test[:, 0, 1].flatten(),  label='Actual Volatility',color='#4A4A90')
    plt.plot(timestamps_test, y_pred_test[:, 0, 1].flatten(),  label='Predicted Volatility',color='#E8AC2F')
    plt.title('Volatility Predictions')
    plt.xlabel('Time')
    plt.ylabel('Volatility')
    plt.legend()

    plt.tight_layout()
    plt.savefig('results/SSE-predictions-FinGAN.png')
    plt.show()
