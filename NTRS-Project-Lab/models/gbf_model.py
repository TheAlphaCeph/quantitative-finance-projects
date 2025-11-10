from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from .base_model import BaseModel
import numpy as np
import pandas as pd
import optuna
from .utils import winsorize_series

class GBFModel(BaseModel):
    """Gradient Boosting with Optuna optimization (using pandas only)"""

    _param_cache = {}
    
    def __init__(self):
        super().__init__("gbf")
        self.model = HistGradientBoostingRegressor(
            learning_rate=0.1,
            max_iter=100,
            loss='absolute_error',
            random_state=10
        )
        self.param_grid = {
            'max_depth': [3, 5, 7, 9],
            'min_samples_leaf': [50, 75, 100, 125]
        }

    def _convert_to_numpy(self, data):
        """Convert data to a NumPy array of type float32."""
        if isinstance(data, np.ndarray):
            return data.astype(np.float32)
        elif hasattr(data, 'values'):
            return data.values.astype(np.float32)
        else:
            return np.array(data, dtype=np.float32)

    def fit(self, train_data):
        
        # Winsorization processing
        train_data['csho_winsorized'] = winsorize_series(train_data['csho'])
        
        # Convert to NumPy and perform scaling
        csho_vals = self._convert_to_numpy(train_data['csho_winsorized'])
        X_train = self._convert_to_numpy(train_data[self.input_list]) / csho_vals[:, None]
        y_train = self._convert_to_numpy(train_data['E_future']) / csho_vals

        # Optuna parameter tuning
        year_hash = hash(str(train_data['year'].unique()))
        if year_hash not in self._param_cache:
            study = optuna.create_study(direction='minimize')
            study.optimize(
                lambda trial: self._objective(trial, X_train, y_train),
                n_trials=30,
                n_jobs=-1
            )
            self._param_cache[year_hash] = study.best_params

        # Train using the best parameters
        self.model.set_params(**self._param_cache[year_hash])
        self.model.fit(X_train, y_train)

    def _objective(self, trial, X, y):
        """Objective function for time series optimization."""
        params = {
            'max_depth': trial.suggest_categorical('max_depth', self.param_grid['max_depth']),
            'min_samples_leaf': trial.suggest_categorical('min_samples_leaf', self.param_grid['min_samples_leaf'])
        }
        
        model = HistGradientBoostingRegressor(
            learning_rate=0.1,
            max_iter=100, 
            loss='absolute_error',
            random_state=10,
            **params
        )
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_train, y_train)
            scores.append(-model.score(X_val, y_val))  
            
        return np.mean(scores)

    def predict(self, test_data):
        # Assumes test_data is a pandas DataFrame
        
        # Data preprocessing
        test_data['csho_winsorized'] = winsorize_series(test_data['csho'])
        
        # Convert to NumPy
        csho_vals = self._convert_to_numpy(test_data['csho_winsorized'])
        X_test = self._convert_to_numpy(test_data[self.input_list]) / csho_vals[:, None]
        
        # Prediction
        y_pred_scaled = self.model.predict(X_test)
        y_pred_unscaled = y_pred_scaled * csho_vals
        
        return pd.Series(y_pred_unscaled, index=test_data.index)