import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from .base_model import BaseModel
from .utils import winsorize_series

class RidgeModel(BaseModel):
    """Ridge forecasting model."""

    def __init__(self):
        super().__init__("ridge")
        alphas_ridge = np.linspace(5e1, 1e3, 500)
        tscv = TimeSeriesSplit(n_splits=5)
        self.model = RidgeCV(alphas=alphas_ridge, fit_intercept=False, cv=tscv)

    def fit(self, train_data):
        train_data['csho_winsorized'] = winsorize_series(train_data['csho'])
        X_train = train_data[self.input_list].div(train_data['csho_winsorized'], axis=0)
        y_train = train_data['E_future'].div(train_data['csho_winsorized'], axis=0)
        self.model.fit(X_train, y_train)

    def predict(self, test_data):
        test_data['csho_winsorized'] = winsorize_series(test_data['csho'])
        X_test = test_data[self.input_list].div(test_data['csho_winsorized'], axis=0)
        y_pred_scaled = self.model.predict(X_test)
        y_pred_unscaled = y_pred_scaled * test_data['csho_winsorized'].values
        return y_pred_unscaled