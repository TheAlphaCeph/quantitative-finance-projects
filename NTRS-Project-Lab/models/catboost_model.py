import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
#from sklearn.metrics import r2_score

from catboost import CatBoostRegressor

from models.base_model import BaseModel
from models.utils import winsorize_series


class CatBoostModel(BaseModel):


    _param_cache = {}

    def __init__(self):
        super().__init__("catboost")
        # Default CatBoostRegressor
        self.model = CatBoostRegressor(
            iterations=500,
            learning_rate=0.1,
            depth=6,
            random_seed=10,
            verbose=False
        )
        # Example hyperparameter search space for GridSearchCV
        self.param_grid = {
            'depth': [3, 5, 7, 9],
            'subsample': [0.6, 0.8, 1.0],
        }

    def _preprocess(self, data: pd.DataFrame) -> pd.DataFrame:

        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        # Winsorize 'csho' to avoid extreme outliers
        data['csho_winsorized'] = winsorize_series(data['csho'])

        # Divide each input feature by 'csho_winsorized'
        for col in self.input_list:
            data[col] = data[col] / data['csho_winsorized']

        return data

    def fit(self, train_data: pd.DataFrame):
        """Train the model with grid search for hyperparameter tuning."""
        train_data = self._preprocess(train_data)
        X_train = train_data[self.input_list].values
        y_train = (train_data['E_future'] / train_data['csho_winsorized']).values

        # Hash of unique 'year' values to handle parameter caching
        year_hash = hash(str(train_data['year'].unique()))

        if year_hash not in self._param_cache:
            # Use GridSearchCV for hyperparameter tuning
            tss = TimeSeriesSplit(n_splits=5)
            grid = GridSearchCV(
                estimator=CatBoostRegressor(
                    iterations=100,
                    random_seed=10,
                    verbose=False
                ),
                param_grid=self.param_grid,
                scoring='r2',
                cv=tss,
                n_jobs=-1  # use all CPU cores
            )
            grid.fit(X_train, y_train)
            self._param_cache[year_hash] = grid.best_params_

        # Rebuild final model with best-found parameters
        best_params = self._param_cache[year_hash]
        self.model = CatBoostRegressor(
            iterations=500,
            random_seed=10,
            verbose=False,
            **best_params
        )
        self.model.fit(X_train, y_train)

    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        """Predict on test data, scale up by 'csho_winsorized'."""
        test_data = self._preprocess(test_data)
        X_test = test_data[self.input_list].values
        y_pred_scaled = self.model.predict(X_test)
        return y_pred_scaled * test_data['csho_winsorized'].values
