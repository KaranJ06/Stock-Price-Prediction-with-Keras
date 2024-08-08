import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
import json
from datetime import datetime, timedelta
import os
from typing import List

# Function to get stock data (from CSV if exists, otherwise from yfinance)
def get_stock_data(ticker, csv_filename):
    if os.path.exists(csv_filename):
        print(f"Reading data from {csv_filename}")
        return pd.read_csv(csv_filename, parse_dates=['Date'])
    else:
        print(f"Fetching data from yfinance API for {ticker}")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5 * 365)  # 5 years of data
        data = yf.download(ticker, start=start_date, end=end_date)
        data = data.reset_index()
        data.to_csv(csv_filename, index=False)
        return data

# Prepare the data
def prepare_stock_data(stock_data):
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data = stock_data.sort_values('Date')
    return stock_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

# Function to create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length, 3])  # Predicting the 'Close' price
    return np.array(X), np.array(y)

# Define the function to create and train the model
def lstm_stock_model(learning_rate, batch_size, dropout_rate):
    model = Sequential([
        Input(shape=(7, 5)),  # Explicitly define input shape
        LSTM(256, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(128),
        Dropout(dropout_rate),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=int(batch_size),
        validation_split=0.2,
        verbose=0,
        callbacks=[EarlyStopping(patience=10), ReduceLROnPlateau(patience=5)]
    )

    # Return negative MSE as we want to maximize this value
    return -history.history['val_loss'][-1]

def load_or_optimize_params():
    params_path = 'best_params.json'
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            params = json.load(f)
        print("Loaded existing parameters")
        return params
    else:
        print("No existing parameters found. Running optimization...")
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

        optimizer.maximize(init_points=5, n_iter=5)

        best_params = optimizer.max['params']

        # Save the optimized parameters
        with open(params_path, 'w') as f:
            json.dump(best_params, f)

        return best_params

def create_model(input_shape, learning_rate, dropout_rate):
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

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"Evaluation Results - MSE: {mse}, MAE: {mae}, R2: {r2}")

    # Plotting true vs predicted values
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='True Values')
    plt.plot(predictions, label='Predicted Values')
    plt.title('True vs Predicted Stock Prices')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

def train_and_predict(tickers, continue_training=False):
    global X_train, y_train  # Make these global so they can be used in lstm_stock_model
    all_X_train, all_y_train = [], []
    all_X_test, all_y_test = [], []
    scalers = {}
    stock_data_dict = {}

    for ticker in tickers:
        csv_filename = f"Data_csv/{ticker}_stock_data.csv"
        data = get_stock_data(ticker, csv_filename)
        stock_data = prepare_stock_data(data)
        stock_data_dict[ticker] = stock_data

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(stock_data.iloc[:, 1:].values)  # Exclude 'Date' column
        scalers[ticker] = scaler

        X, y = create_sequences(scaled_data, 7)

        split = int(0.8 * len(X))
        all_X_train.append(X[:split])
        all_X_test.append(X[split:])
        all_y_train.append(y[:split])
        all_y_test.append(y[split:])

    X_train = np.concatenate(all_X_train)
    y_train = np.concatenate(all_y_train)
    X_test = np.concatenate(all_X_test)
    y_test = np.concatenate(all_y_test)

    best_params = load_or_optimize_params()

    if continue_training and os.path.exists('stock_model.keras'):
        model = load_model('stock_model.keras')
        print("Loaded existing model")
    else:
        model = create_model((7, 5), best_params['learning_rate'], best_params['dropout_rate'])
        print("Created new model")

    model.fit(X_train, y_train, epochs=2, batch_size=int(best_params['batch_size']), validation_split=0.2,
              callbacks=[EarlyStopping(patience=10), ReduceLROnPlateau(patience=5)])

    model.save('stock_model.keras')

    evaluate_model(model, X_test, y_test)

    for ticker in tickers:
        # Predict next month (assuming 30 trading days)
        last_sequence = scalers[ticker].transform(stock_data_dict[ticker].iloc[-7:, 1:].values)
        next_month_predictions = []

        for _ in range(30):
            next_day_scaled = model.predict(last_sequence.reshape(1, 7, 5))
            next_month_predictions.append(next_day_scaled[0, 0])

            # Update the last sequence
            last_sequence = np.roll(last_sequence, -2, axis=0)
            last_sequence[-1, 3] = next_day_scaled

        # Convert predictions to actual prices
        dummy_pred = np.zeros((len(next_month_predictions), 5))
        dummy_pred[:, 3] = next_month_predictions
        next_month_prices = scalers[ticker].inverse_transform(dummy_pred)[:, 3]

        print(f"Predicted prices for {ticker} next month:")
        for day, price in enumerate(next_month_prices, 1):
            print(f"Day {day}: ${price:.2f}")

        # Plotting future predictions
        plt.figure(figsize=(16, 8))
        plt.plot(range(1, 31), next_month_prices, marker='o')
        plt.title(f'{ticker} Stock Price Prediction for Next Month')
        plt.xlabel('Days')
        plt.ylabel('Predicted Price ($)')
        plt.grid(True)

        # Annotate the first and last predicted prices
        plt.annotate(f'${next_month_prices[0]:.2f}', (1, next_month_prices[0]), textcoords="offset points",
                     xytext=(0, 10), ha='center')
        plt.annotate(f'${next_month_prices[-1]:.2f}', (30, next_month_prices[-1]), textcoords="offset points",
                     xytext=(0, -15), ha='center')

        # Add overall trend
        z = np.polyfit(range(1, 31), next_month_prices, 1)
        p = np.poly1d(z)
        plt.plot(range(1, 31), p(range(1, 31)), "r--", alpha=0.8, label='Trend')

        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{ticker}_future_price_prediction_plot.png')
        plt.close()

# Main execution
if __name__ == "__main__":
    tickers = ["^GSPC", "AAPL", "GOOG", "MSFT", "NVDA", "META", "AMZN", "TSLA", "NFLX", "BABA", "V"]  # List of tickers to train on
    continue_training = True  # Set to True if you want to continue training existing model

    train_and_predict(tickers, continue_training)