import json
from bayes_opt import BayesianOptimization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scripts.model import create_model
from scripts.utils import load_config
from scripts.stock_model import X_train, y_train  # Import global variables
import os
def lstm_stock_model(learning_rate, batch_size, dropout_rate):
    model = create_model((7, 5), learning_rate, dropout_rate)
    history = model.fit(
        X_train, y_train,
        epochs=load_config()['epochs_optimizer'],
        batch_size=int(batch_size),
        validation_split=0.2,
        verbose=0,
        callbacks=[EarlyStopping(patience=10), ReduceLROnPlateau(patience=5)]
    )
    return -history.history['val_loss'][-1]

def load_or_optimize_params():
    params_path = 'scripts/best_params.json'
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

        optimizer.maximize(init_points=load_config()['init_points'], n_iter=load_config()['n_iter'])

        best_params = optimizer.max['params']

        with open(params_path, 'w') as f:
            json.dump(best_params, f)

        return best_params
