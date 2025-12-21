from cmath import sqrt

import keras.initializers.initializers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Concatenate, Flatten, Reshape, LeakyReLU, Dropout, GRU
from keras.optimizers import Adam, RMSprop

def prepare_data(file_path, window_size=30, forecast_horizon=5):
    df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')

    # 计算收益率和波动率
    df['return'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['return'].rolling(window=20).std() * np.sqrt(240)
    df.dropna(inplace=True)

    features = ['return', 'volatility', 'volume']
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


# 修改后的FinGAN类支持消融实验配置
class FinGAN:
    def __init__(self, window_size, feature_dim, target_dim, forecast_horizon,
                 gen_lstm=True, dis_dropout=True, use_leakyrelu=True,
                 adversarial=True, rnn_type='lstm'):
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.target_dim = target_dim
        self.forecast_horizon = forecast_horizon
        self.gen_lstm = gen_lstm
        self.dis_dropout = dis_dropout
        self.use_leakyrelu = use_leakyrelu
        self.adversarial = adversarial
        self.rnn_type = rnn_type.lower()

        # 构建模型组件
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator() if adversarial else None
        self.combined = self.build_combined() if adversarial else None

    def build_generator(self):
        initializer = keras.initializers.initializers.GlorotNormal()
        input_layer = Input(shape=(self.window_size, self.feature_dim))

        # RNN层选择
        if self.gen_lstm:
            if self.rnn_type == 'lstm':
                x = LSTM(256, return_sequences=True, kernel_initializer=initializer)(input_layer)
                x = LSTM(128, kernel_initializer=initializer)(x)
            elif self.rnn_type == 'gru':
                x = GRU(256, return_sequences=True, kernel_initializer=initializer)(input_layer)
                x = GRU(128, kernel_initializer=initializer)(x)
        else:
            x = Flatten()(input_layer)
            x = Dense(256, activation='relu')(x)
            x = Dense(128, activation='relu')(x)

        x = Dense(256, activation='relu', kernel_initializer=initializer)(x)
        output = Dense(self.forecast_horizon * self.target_dim, activation='tanh')(x)
        output = Reshape((self.forecast_horizon, self.target_dim))(output)
        return Model(input_layer, output)

    def build_discriminator(self):
        initializer = keras.initializers.initializers.GlorotNormal()
        hist_input = Input(shape=(self.window_size, self.feature_dim))
        target_input = Input(shape=(self.forecast_horizon, self.target_dim))

        # 历史数据处理
        if self.rnn_type == 'lstm':
            x = LSTM(128, kernel_initializer=initializer)(hist_input)
        elif self.rnn_type == 'gru':
            x = GRU(128, kernel_initializer=initializer)(hist_input)
        else:
            x = Flatten()(hist_input)
            x = Dense(128, activation='relu')(x)

        y = Flatten()(target_input)
        combined = Concatenate()([x, y])

        x = Dense(256, kernel_initializer=initializer)(combined)
        x = LeakyReLU(0.2)(x) if self.use_leakyrelu else Dense(256, activation='relu')(x)
        if self.dis_dropout: x = Dropout(0.1)(x)

        x = Dense(128, kernel_initializer=initializer)(x)
        x = LeakyReLU(0.2)(x) if self.use_leakyrelu else Dense(128, activation='relu')(x)
        if self.dis_dropout: x = Dropout(0.1)(x)

        validity = Dense(1, activation='sigmoid')(x)
        return Model([hist_input, target_input], validity)

    def build_combined(self):
        self.discriminator.trainable = False
        hist_input = Input(shape=(self.window_size, self.feature_dim))
        generated_target = self.generator(hist_input)
        validity = self.discriminator([hist_input, generated_target])
        return Model(hist_input, [validity, generated_target])

    def train(self, X_train, y_train, epochs=200, batch_size=32):
        if self.adversarial:
            # GAN训练模式
            self.discriminator.compile(loss='binary_crossentropy',
                                       optimizer=RMSprop(1e-4), metrics=['accuracy'])
            self.combined.compile(loss=['binary_crossentropy', 'mse'],
                                  loss_weights=[1, 100],
                                  optimizer=RMSprop(1e-4))

            real = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))

            for epoch in range(epochs):
                # 判别器训练
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                real_seqs = X_train[idx]
                real_targets = y_train[idx]
                fake_targets = self.generator.predict(real_seqs, verbose=0)

                d_loss_real = self.discriminator.train_on_batch([real_seqs, real_targets], real)
                d_loss_fake = self.discriminator.train_on_batch([real_seqs, fake_targets], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # 生成器训练
                g_loss = self.combined.train_on_batch(real_seqs, [real, real_targets])

                #if epoch % 10 == 0:
                print(f"Epoch {epoch} [D loss: {d_loss[0]:.4f} acc: {d_loss[1]:.2f}] [G loss: {g_loss[0]:.4f}]")
        else:
            # 纯监督学习模式
            self.generator.compile(loss='mse', optimizer=RMSprop(1e-4))
            self.generator.fit(X_train, y_train,
                               epochs=epochs,
                               batch_size=batch_size,
                               verbose=0)


# 评估函数
def evaluate_model(model, X_test, y_test, scaler_y, timestamps, config_name):
    y_pred_scaled = model.generator.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, y_test.shape[2])).reshape(y_pred_scaled.shape)
    y_real = scaler_y.inverse_transform(y_test.reshape(-1, y_test.shape[2])).reshape(y_test.shape)

    test_size = int(len(y_test) * 0.2)
    y_real_test = y_real[-test_size:]
    y_pred_test = y_pred[-test_size:]

    results = {}
    for i, target in enumerate(['Return', 'Volatility']):
        y_t = y_real_test[:, :, i].flatten()
        y_p = y_pred_test[:, :, i].flatten()

        results[f'{target}_MAE'] = mean_absolute_error(y_t, y_p)
        results[f'{target}_MSE'] = mean_squared_error(y_t, y_p)
        results[f'{target}_RMSE'] =sqrt(mean_squared_error(y_t, y_p))
        results[f'{target}_R2'] = r2_score(y_t, y_p)


    # 绘制结果图
    plt.figure(figsize=(12, 6))
    for i, target in enumerate(['Return', 'Volatility']):
        plt.subplot(2, 1, i + 1)
        plt.plot(y_real_test[:, 0, i], label='Actual')
        plt.plot(y_pred_test[:, 0, i], label='Predicted')
        plt.title(f'{config_name} - {target} Prediction')
        plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/{config_name}_predictions.png')
    plt.close()

    return results


if __name__ == "__main__":
    # 实验配置
    ablation_configs = [
        {'name': 'Baseline', 'gen_lstm': True, 'dis_dropout': True,
         'use_leakyrelu': True, 'adversarial': True, 'rnn_type': 'lstm'},

        {'name': 'No_Adversarial', 'gen_lstm': True, 'dis_dropout': True,
         'use_leakyrelu': True, 'adversarial': False, 'rnn_type': 'lstm'},

        # {'name': 'No_LSTM', 'gen_lstm': False, 'dis_dropout': True,
        #  'use_leakyrelu': True, 'adversarial': True, 'rnn_type': 'dense'},

        {'name': 'GRU_Replace', 'gen_lstm': True, 'dis_dropout': True,
         'use_leakyrelu': True, 'adversarial': True, 'rnn_type': 'gru'},

        # {'name': 'No_Dropout', 'gen_lstm': True, 'dis_dropout': False,
        #  'use_leakyrelu': True, 'adversarial': True, 'rnn_type': 'lstm'},

        {'name': 'ReLU_Only', 'gen_lstm': True, 'dis_dropout': True,
         'use_leakyrelu': False, 'adversarial': True, 'rnn_type': 'lstm'}
    ]

    # 初始化参数
    WINDOW_SIZE = 30
    FORECAST_HORIZON = 5
    EPOCHS = 200  # 为演示缩短训练轮次
    BATCH_SIZE = 128

    # 准备数据
    X, y, scaler_X, scaler_y, timestamps = prepare_data('data/SP500.csv',
                                                        window_size=WINDOW_SIZE,
                                                        forecast_horizon=FORECAST_HORIZON)

    # 运行消融实验
    all_results = {}
    for config in ablation_configs:
        print(f"\n=== Running Experiment: {config['name']} ===")

        model = FinGAN(
            window_size=WINDOW_SIZE,
            feature_dim=X.shape[2],
            target_dim=y.shape[2],
            forecast_horizon=FORECAST_HORIZON,
            **{k: v for k, v in config.items() if k != 'name'}
        )

        model.train(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE)
        results = evaluate_model(model, X, y, scaler_y, timestamps, config['name'])
        all_results[config['name']] = results

    # 打印对比结果
    print("\n=== Ablation Study Results ===")
    metrics = ['Return_MAE','Return_MSE','Return_RMSE', 'Return_R2', 'Volatility_MAE','Volatility_MSE','Volatility_RMSE', 'Volatility_R2']
    print(f"{'Model':<15}", end="")
    for m in metrics: print(f"{m:<15}", end="")
    print()

    for model_name, res in all_results.items():
        print(f"{model_name:<15}", end="")
        for m in metrics:
            print(f"{res[m]:<15.4f}", end="")
        print()

    # 可视化对比
    plt.figure(figsize=(12, 8))
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i + 1)
        values = [res[metric] for res in all_results.values()]
        plt.bar(all_results.keys(), values)
        plt.title(metric)
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/ablation_comparison.png')
    plt.show()