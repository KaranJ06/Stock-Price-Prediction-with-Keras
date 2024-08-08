from scripts.stock_model import StockModel
from scripts.utils import load_config

if __name__ == "__main__":
    tickers = load_config()["tikets"]
    continue_training = True

    stock_model = StockModel(tickers, continue_training)
    stock_model.train_and_evaluate()