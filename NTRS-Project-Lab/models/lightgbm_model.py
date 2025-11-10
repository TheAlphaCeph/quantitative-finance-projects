import numpy as np
import lightgbm as lgb
import pandas as pd
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from .base_model import BaseModel
from .utils import winsorize_series

class LightGBMModel(BaseModel):
    """LightGBM forecasting model with hyperparameter tuning & walk-forward validation."""

    def __init__(self):
        super().__init__("lightgbm")
        self.model = None  
        self.best_params_cache = {}  # Cache best hyperparameters

    def _objective(self, trial, X_train, y_train):
        """Optuna objective function for hyperparameter tuning using TimeSeriesSplit."""
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.1),
            "num_leaves": trial.suggest_int("num_leaves", 20, 80),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.6, 1.0),
            "subsample": trial.suggest_uniform("subsample", 0.6, 1.0),
            "n_estimators": 500
        }

        tscv = TimeSeriesSplit(n_splits=5)
        rmse_list = []

        for train_index, valid_index in tscv.split(X_train):
            X_train_fold, X_valid_fold = X_train.iloc[train_index].values, X_train.iloc[valid_index].values
            y_train_fold, y_valid_fold = y_train.iloc[train_index].values, y_train.iloc[valid_index].values

            model = lgb.LGBMRegressor(**params)
            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_valid_fold)
            rmse = mean_squared_error(y_valid_fold, y_pred, squared=False)
            rmse_list.append(rmse)

        return np.mean(rmse_list)  # Average RMSE across all splits

    def fit(self, train_data):
        """Fits LightGBM model with hyperparameter tuning"""

        train_data['csho_winsorized'] = winsorize_series(train_data['csho'])

        X_train = train_data[self.input_list].div(train_data['csho_winsorized'], axis=0)
        y_train = train_data['E_future'].div(train_data['csho_winsorized'], axis=0)

        # Only tune hyperparameters if they haven't been set before
        if not self.best_params_cache:
            print(f"Tuning hyperparameters using TimeSeriesSplit...")
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: self._objective(trial, X_train, y_train), n_trials=10)
            self.best_params_cache = study.best_params  # Cache best parameters

        best_params = self.best_params_cache
        print(f"Best Parameters: {best_params}")

        # Train final model with best parameters
        self.model = lgb.LGBMRegressor(**best_params, n_estimators=500)
        self.model.fit(X_train, y_train)

    def predict(self, test_data):
        """Generates predictions using the trained LightGBM model."""

        test_data['csho_winsorized'] = winsorize_series(test_data['csho'])

        # Scale features by `csho_winsorized`
        X_test = test_data[self.input_list].div(test_data['csho_winsorized'], axis=0)

        y_pred_scaled = self.model.predict(X_test)

        # Convert back to total earnings
        y_pred_unscaled = y_pred_scaled * test_data['csho_winsorized'].values
        return y_pred_unscaled  