
# README for Stock Price Prediction Project

## Project Overview

This project aims to predict future stock prices using historical data and a Long Short-Term Memory (LSTM) model. The pipeline includes data acquisition, preprocessing, model training, evaluation, and visualization. The predictions are tailored for individual stocks, with detailed analysis provided at each stage.

**LSTM model structure:**

![image](https://github.com/user-attachments/assets/321bac04-d100-4313-b39c-09f4da481cda)
## Project Structure

- **data/**: Contains CSV files for each stock ticker with historical data.
- **models/**: Stores the trained Keras model (`stock_model.keras`).
- **predictions/**: Stores visualizations and prediction results for individual stocks.
- **scripts/**: Contains configuration files and Python scripts for data processing, model training, and analysis.

## Dependencies

- **Python 3.x**
- Required Python packages: `numpy`, `pandas`, `yfinance`, `matplotlib`, `seaborn`, `scikit-learn`, `keras`, `bayes_opt`

Install all dependencies using the following command:

```bash
pip install -r requirements.txt
```

# Project Configuration and Workflow

## Configuration

The configuration file (`scripts/config.json`) contains key parameters:

- **tikets**: List of stock tickers to analyze.
- **epochs**: Number of epochs for training the model.
- **train_split**: Ratio of the dataset used for training.
- **validation_split**: Ratio of the training set used for validation.
- **seq_length**: Length of the input sequence for the LSTM model.
- **days_to_predict**: Number of days to predict future stock prices.
- **init_points** and **n_iter**: Parameters for Bayesian optimization.

## Workflow

### Data Acquisition and Preparation

- Stock data is fetched from the Yahoo Finance API or loaded from CSV files if already available.
- Data is preprocessed by sorting dates, normalizing features, and creating input sequences for the LSTM model.

### Statistical Analysis

- **Statistical Summary**: Understand the general behavior of each stock (e.g., average prices, volatility).
- **Correlation Analysis**: Create a correlation matrix to understand relationships between features.
- **Data Exploration**: Visualize stock price trends over time.
- **Volatility Analysis**: Analyze rolling volatility to gauge risk.

### Model Training

- An LSTM model is built and trained on the prepared data.
- Bayesian optimization is used to find the best hyperparameters.
- The trained model is saved for future use.

### Evaluation

- Model performance is evaluated using Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R2).
- True vs. Predicted stock prices are visualized.
- Error distributions are plotted to understand prediction accuracy.

### Future Predictions

- The model predicts future stock prices for the next `n` days.
- Results are visualized and saved for each stock.

## Running the Project

To run the project, execute the main script:

```bash
python main.py
```

## Script Overview

This script will:

- Load the stock tickers from the configuration file.
- Summarize and analyze the data.
- Train and evaluate the LSTM model.
- Save and visualize predictions.

## Logging

Logging is set up to capture detailed information about the execution process, which is helpful for debugging and understanding model behavior.

## Visualization

Several visualization functions are provided:

- **Correlation Matrix**: Displays relationships between features.
- **Stock Price Over Time**: Shows historical stock prices.
- **Rolling Volatility**: Illustrates the volatility of stock prices.
- **True vs. Predicted Prices**: Compares actual and predicted stock prices.
- **Error Distribution**: Analyzes prediction errors.
- **Future Price Prediction**: Projects future stock prices.

These visualizations help in selecting stocks and understanding the model's predictions.

## Conclusion

This project provides a comprehensive pipeline for stock price prediction using LSTM models. The detailed analysis and visualization offer insights into stock behavior, aiding in better investment decisions.
