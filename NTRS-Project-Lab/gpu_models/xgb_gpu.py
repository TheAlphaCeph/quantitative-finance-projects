from xgboost import XGBRegressor
from .base_model import BaseModel
from .utils import winsorize_series
import optuna
import numpy as np
import cudf
import cupy as cp
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit

class XGBModel(BaseModel):
    """Optimized XGBoost with GPU acceleration"""
    
    _param_cache = {}
    
    def __init__(self):
        super().__init__("xgb")
        self.model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0,
            tree_method='hist',
            device='cuda',
            random_state=10
        )
        self.param_grid = {
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2]
        }

    def _gpu_preprocess(self, data):
        """GPU accelerated preprocessing (cuDF compatible)"""
        if not isinstance(data, cudf.DataFrame):
            data = cudf.from_pandas(data)
        data['csho_winsorized'] = winsorize_series(data['csho'].to_pandas()).values
        for col in self.input_list:
            data[col] = data[col] / data['csho_winsorized']
        return data

    def fit(self, train_data):
        train_data = self._gpu_preprocess(train_data)
        X_train = train_data[self.input_list].values
        y_train = (train_data['E_future'] / train_data['csho_winsorized']).values

        year_hash = hash(str(train_data['year'].unique()))
        if year_hash not in self._param_cache:
            study = optuna.create_study(direction='maximize')
            study.optimize(
                lambda trial: self._objective(trial, X_train, y_train),
                n_trials=50,
                n_jobs=-1
            )
            self._param_cache[year_hash] = study.best_params

        params = self._param_cache[year_hash]
        self.model = XGBRegressor(
            n_estimators=500,
            tree_method='hist',
            device='cuda',
            random_state=10,
            **params
        )
        self.model.fit(X_train, y_train)

    def _objective(self, trial, X, y):
        """Optuna optimization objective function with time series split"""
        params = {
            'max_depth': trial.suggest_categorical('max_depth', self.param_grid['max_depth']),
            'learning_rate': trial.suggest_categorical('learning_rate', self.param_grid['learning_rate']),
            'subsample': trial.suggest_categorical('subsample', self.param_grid['subsample']),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', self.param_grid['colsample_bytree']),
            'gamma': trial.suggest_categorical('gamma', self.param_grid['gamma'])
        }
        
        model = XGBRegressor(
            n_estimators=100,
            tree_method='hist',
            device='cuda',
            **params,
            random_state=10
        )
        
        # Convert GPU arrays to CPU for compatibility with sklearn
        if isinstance(X, cp.ndarray):
            X = X.get()
        if isinstance(y, cp.ndarray):
            y = y.get()
        
        # Time Series Cross-Validation (3 splits)
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            scores.append(r2_score(y_val, y_pred))
        
        # Return average validation score
        return np.mean(scores)

    def predict(self, test_data):
        test_data = self._gpu_preprocess(test_data)
        X_test = test_data[self.input_list].values
        y_pred_scaled = self.model.predict(X_test)
        result = y_pred_scaled * test_data['csho_winsorized'].values.get()
        
        return result