#from cuml.ensemble import RandomForestRegressor as cuRFRegressor
from sklearn.model_selection import TimeSeriesSplit
#import cupy as cp
import numpy as np
#import cudf
import pandas as pd
import optuna
from .base_model import BaseModel
from .utils import winsorize_series

class RFModel(BaseModel):
    """GPU-accelerated Random Forest with Optuna optimization and TimeSeriesSplit CV"""

    _param_cache = {}
    
    def __init__(self):
        super().__init__("rf")
        self.model = cuRFRegressor(
            n_estimators=500,
            split_criterion='mse',  
            max_features='sqrt',
            max_depth=20,
            min_samples_leaf=15,
            random_state=10
        )
        self.param_grid = {
            'max_features': ['sqrt'],
            'max_depth': [20, 25, 30, 35],
            'min_samples_leaf': [15, 20, 25, 50]
        }
    
    def _convert_to_numpy(self, data):
        """Convert GPU (cuDF) data to NumPy arrays if necessary."""
        if isinstance(data, cudf.DataFrame):
            return data.to_pandas().values.astype(np.float32)
        elif isinstance(data, cudf.Series):
            return data.to_pandas().values.astype(np.float32)
        elif isinstance(data, np.ndarray):
            return data.astype(np.float32)
        elif hasattr(data, 'values'):
            return data.values.astype(np.float32)
        else:
            return np.array(data, dtype=np.float32)
    
    def fit(self, train_data):
        if isinstance(train_data, cudf.DataFrame):
            train_data = train_data.to_pandas()
        
        train_data['csho_winsorized'] = winsorize_series(train_data['csho'])
        
        csho_vals = self._convert_to_numpy(train_data['csho_winsorized'])
        X_train = self._convert_to_numpy(train_data[self.input_list]) / csho_vals[:, None]
        y_train = self._convert_to_numpy(train_data['E_future']) / csho_vals

        year_hash = hash(str(train_data['year'].unique()))
        if year_hash not in self._param_cache:
            study = optuna.create_study(direction='maximize')
            study.optimize(
                lambda trial: self._objective(trial, X_train, y_train),
                n_trials=30,
                n_jobs=-1
            )
            self._param_cache[year_hash] = study.best_params

        best_params = self._param_cache[year_hash]
        self.model = cuRFRegressor(
            n_estimators=500,
            split_criterion='mse',
            max_features=best_params['max_features'],
            max_depth=best_params['max_depth'],
            min_samples_leaf=best_params['min_samples_leaf'],
            random_state=10
        )
        X_train_gpu = cp.asarray(X_train)
        y_train_gpu = cp.asarray(y_train)
        self.model.fit(X_train_gpu, y_train_gpu)
    
    def _objective(self, trial, X, y):
        """Objective function with TimeSeriesSplit cross-validation for tuning."""
        params = {
            'max_features': trial.suggest_categorical('max_features', self.param_grid['max_features']),
            'max_depth': trial.suggest_categorical('max_depth', self.param_grid['max_depth']),
            'min_samples_leaf': trial.suggest_categorical('min_samples_leaf', self.param_grid['min_samples_leaf'])
        }
        
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            X_tr_gpu = cp.asarray(X_tr)
            y_tr_gpu = cp.asarray(y_tr)
            X_val_gpu = cp.asarray(X_val)
            y_val_gpu = cp.asarray(y_val)
            
            model = cuRFRegressor(
                n_estimators=500,
                split_criterion='mse',
                max_features=params['max_features'],
                max_depth=params['max_depth'],
                min_samples_leaf=params['min_samples_leaf'],
                random_state=10
            )
            model.fit(X_tr_gpu, y_tr_gpu)
            y_pred_gpu = model.predict(X_val_gpu)
            y_pred = cp.asnumpy(y_pred_gpu)
            score = -np.mean((y_val - y_pred) ** 2)
            scores.append(score)
        
        return np.mean(scores)
    
    def predict(self, test_data):
        if isinstance(test_data, cudf.DataFrame):
            test_data = test_data.to_pandas()
        
        test_data['csho_winsorized'] = winsorize_series(test_data['csho'])
        
        csho_vals = self._convert_to_numpy(test_data['csho_winsorized'])
        X_test = self._convert_to_numpy(test_data[self.input_list]) / csho_vals[:, None]
        X_test_gpu = cp.asarray(X_test)
        y_pred_gpu = self.model.predict(X_test_gpu)
        y_pred = cp.asnumpy(y_pred_gpu)
        y_pred_unscaled = y_pred * csho_vals
        
        return pd.Series(y_pred_unscaled, index=test_data.index)