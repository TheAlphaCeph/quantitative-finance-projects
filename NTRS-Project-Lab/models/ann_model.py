import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from .base_model import BaseModel
from .utils import winsorize_series

class ANNBaggingModel(BaseModel):
    """Artificial Neural Network model with Bootstrap Aggregating using Scikit-Learn."""

    def __init__(self, use_params=False):
        super().__init__("ann_bagging")
        self.use_params = use_params
        self.fixed_parameters = {
            'activation': 'relu',
            'hidden_layer_sizes': (8, 4),
            'alpha': 1e-3
        }
        
        if self.use_params:
            self.regr = MLPRegressor(max_iter=1000, random_state=10, early_stopping=True, tol=1e-6, **self.fixed_parameters)
        else:
            self.parameters = {
                'activation': ['relu', 'tanh'],
                'hidden_layer_sizes': [(64, 32, 16, 8), (32, 16, 8, 4), (16, 8, 4, 2),
                                       (64, 32, 16), (32, 16, 8), (16, 8, 4), (8, 4, 2),
                                       (64, 32), (32, 16), (16, 8), (8, 4), (4, 2),
                                       (64,), (32,), (16,), (8,), (4,)],
                'alpha': [1e-3, 1e-4, 1e-5]
            }
            tss = TimeSeriesSplit(n_splits=5)
            self.regr = GridSearchCV(MLPRegressor(max_iter=1000, random_state=10, early_stopping=True, tol=1e-6),
                                     self.parameters, cv=tss, n_jobs=-1, scoring='neg_mean_squared_error')
        
        self.model = None  # Initialize model as None until trained

    def fit(self, train_data):
        # Calculate winsorized 'csho' series
        train_data['csho_winsorized'] = winsorize_series(train_data['csho'])
        
        X_train = train_data[self.input_list].values / train_data['csho_winsorized'].values[:, None]
        y_train = train_data['E_future'].values / train_data['csho_winsorized'].values
        
        print(f"Data shape: {train_data.shape}")
        if self.use_params:
            print("Using fixed parameters for training.")
            self.regr.fit(X_train, y_train)
            self.model = BaggingRegressor(self.regr, random_state=0, n_estimators=10, max_samples=0.6, n_jobs=-1)
        else:
            print("Running GridSearchCV to find the best parameters...")
            self.regr.fit(X_train, y_train)
            print("Best parameters found:", self.regr.best_params_)
            self.model = BaggingRegressor(self.regr.best_estimator_, random_state=0, n_estimators=10, max_samples=0.6, n_jobs=-1)
        
        self.model.fit(X_train, y_train)
        
    def predict(self, test_data):
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() before predict().")
        
        # Calculate winsorized 'csho' series
        test_data['csho_winsorized'] = winsorize_series(test_data['csho'])
        
        X_test = test_data[self.input_list].values / test_data['csho_winsorized'].values[:, None]
        y_true = test_data['E_future'].values
        
        y_pred_scaled = self.model.predict(X_test)
        y_pred_unscaled = y_pred_scaled * test_data['csho_winsorized'].values
        
        # Calculate MSE and R2 Score
        mse = mean_squared_error(y_true, y_pred_unscaled)
        r2 = r2_score(y_true, y_pred_unscaled)
        print(f"MSE: {mse:.4f}, R2 Score: {r2:.4f}")
        
        return pd.Series(y_pred_unscaled, index=test_data.index)
