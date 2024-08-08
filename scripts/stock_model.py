import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scripts.utils import load_config
from scripts.data_handler import get_stock_data, prepare_stock_data
from scripts.visualization import plot_true_vs_predicted, plot_future_predictions
import os
import json
class StockModel:
    def __init__(self, tickers, continue_training=False):
        self.tickers = tickers
        self.continue_training = continue_training
        self.scalers = {}
        self.stock_data_dict = {}
        self.X_train = np.array([])
        self.y_train = np.array([])
        self.X_test = np.array([])
        self.y_test = np.array([])
        self.best_params = {}

    def create_sequences(self, data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length)])
            y.append(data[i + seq_length, 3])  # Predicting the 'Close' price
        return np.array(X), np.array(y)

    def load_or_optimize_params(self):
        params_path = 'scripts/best_params.json'
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                self.best_params = json.load(f)
            print("Loaded existing parameters")
        else:
            from bayes_opt import BayesianOptimization

            def lstm_stock_model(learning_rate, batch_size, dropout_rate):
                model = self.create_model((7, 5), learning_rate, dropout_rate)
                history = model.fit(
                    self.X_train, self.y_train,
                    epochs=load_config()['epochs'],
                    batch_size=int(batch_size),
                    validation_split=load_config()["validation_split"],
                    verbose=0,
                    callbacks=[EarlyStopping(patience=10), ReduceLROnPlateau(patience=5)]
                )
                return -history.history['val_loss'][-1]

            pbounds = {
                'learning_rate': (1e-4, 1e-2),
                'batch_size': (16, 128),
                'dropout_rate': (0.1, 0.5)
            }

            optimizer = BayesianOptimization(
                f=lstm_stock_model,
                pbounds=pbounds,
                random_state=42,
                verbose=2
            )

            optimizer.maximize(init_points=load_config()['init_points'], n_iter=load_config()['n_iter'])

            self.best_params = optimizer.max['params']

            with open(params_path, 'w') as f:
                json.dump(self.best_params, f)

    def create_model(self, input_shape, learning_rate, dropout_rate):
        model = Sequential([
            Input(shape=input_shape),
            LSTM(256, return_sequences=True),
            Dropout(dropout_rate),
            LSTM(128),
            Dropout(dropout_rate),
            Dense(64, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
        return model

    def prepare_data(self):
        all_X_train, all_y_train = [], []
        all_X_test, all_y_test = [], []
        for ticker in self.tickers:
            csv_filename = f"data/{ticker}_stock_data.csv"
            data = get_stock_data(ticker, csv_filename)
            stock_data = prepare_stock_data(data)
            self.stock_data_dict[ticker] = stock_data

            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(stock_data.iloc[:, 1:].values)
            self.scalers[ticker] = scaler

            X, y = self.create_sequences(scaled_data, load_config()['seq_length'])

            split = int(load_config()["train_split"] * len(X))
            all_X_train.append(X[:split])
            all_X_test.append(X[split:])
            all_y_train.append(y[:split])
            all_y_test.append(y[split:])

        self.X_train = np.concatenate(all_X_train)
        self.y_train = np.concatenate(all_y_train)
        self.X_test = np.concatenate(all_X_test)
        self.y_test = np.concatenate(all_y_test)

    def train_and_evaluate(self):
        self.prepare_data()
        self.load_or_optimize_params()

        if self.continue_training and os.path.exists('models/stock_model.keras'):
            model = load_model('models/stock_model.keras')
            print("Loaded existing model")
        else:
            model = self.create_model((7, 5), self.best_params['learning_rate'], self.best_params['dropout_rate'])
            print("Created new model")

        model.fit(self.X_train, self.y_train, epochs=load_config()['epochs'], batch_size=int(self.best_params['batch_size']), validation_split=load_config()["validation_split"],
                  callbacks=[EarlyStopping(patience=10), ReduceLROnPlateau(patience=5)])
        if not os.path.exists("models/"):
            os.makedirs("models/")
        model.save('models/stock_model.keras')

        self.evaluate_model(model)

    def evaluate_model(self, model):
        predictions = model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, predictions)
        mae = mean_absolute_error(self.y_test, predictions)
        r2 = r2_score(self.y_test, predictions)
        print(f"Evaluation Results - MSE: {mse}, MAE: {mae}, R2: {r2}")

        plot_true_vs_predicted(self.y_test, predictions)

        for ticker in self.tickers:
            last_sequence = self.scalers[ticker].transform(self.stock_data_dict[ticker].iloc[-7:, 1:].values)
            next_month_predictions = []

            for _ in range(load_config()["days_to_predict"]):
                next_day_scaled = model.predict(last_sequence.reshape(1, 7, 5))
                next_month_predictions.append(next_day_scaled[0, 0])
                last_sequence = np.roll(last_sequence, -2, axis=0)
                last_sequence[-1, 3] = next_day_scaled

            dummy_pred = np.zeros((len(next_month_predictions), 5))
            dummy_pred[:, 3] = next_month_predictions
            next_month_prices = self.scalers[ticker].inverse_transform(dummy_pred)[:, 3]

            print(f"Predicted prices for {ticker} days:")
            for day, price in enumerate(next_month_prices, 1):
                print(f"Day {day}: ${price:.2f}")

            plot_future_predictions(next_month_prices, ticker)
