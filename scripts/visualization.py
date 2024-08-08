import matplotlib.pyplot as plt
import os
import numpy as np
from scripts.utils import load_config
days_to_predict = load_config()['days_to_predict']
def plot_true_vs_predicted(y_test, predictions):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='True Values')
    plt.plot(predictions, label='Predicted Values')
    plt.title('True vs Predicted Stock Prices')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
    plt.savefig(f'predictions/val_price_prediction_plot.png')
    plt.close()

def plot_future_predictions(next_prices, ticker):
    if not os.path.exists('../predictions'):
        os.makedirs('../predictions')

    plt.figure(figsize=(16, 8))
    plt.plot(range(1, load_config()['days_to_predict']+1), next_prices, marker='o')
    plt.title(f'{ticker} Stock Price Prediction for Next {days_to_predict} days')
    plt.xlabel('Days')
    plt.ylabel('Predicted Price ($)')
    plt.grid(True)
    plt.annotate(f'${next_prices[0]:.2f}', (1, next_prices[0]), textcoords="offset points",
                 xytext=(0, 10), ha='center')
    plt.annotate(f'${next_prices[-1]:.2f}', (days_to_predict, next_prices[-1]), textcoords="offset points",
                 xytext=(0, -15), ha='center')
    z = np.polyfit(range(1, days_to_predict+1), next_prices, 1)
    p = np.poly1d(z)
    plt.plot(range(1, days_to_predict+1), p(range(1, days_to_predict+1)), "r--", alpha=0.8, label='Trend')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'predictions/{ticker}_future_price_prediction_plot.png')
    plt.close()
