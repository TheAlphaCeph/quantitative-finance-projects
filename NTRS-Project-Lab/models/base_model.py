from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from .utils import get_input_list
#import cudf

class BaseModel(ABC):
    """Abstract base class for forecasting models."""

    def __init__(self, name):
        self.name = name
        self.input_list = get_input_list()

    @abstractmethod
    def fit(self, train_data):
        """Fits the model to the training data."""
        pass

    @abstractmethod
    def predict(self, test_data):
        """Generates predictions for the test data."""
        pass

    def run_forecast(self, df, start_year=1975, end_year=2024):
        """Runs the forecasting loop for a single model."""
        all_results = []

        for t in range(start_year, end_year + 1):
            print(t)
            train_mask = (df['year'] >= t - 10) & (df['year'] <= t - 1)
            test_mask = (df['year'] == t)

            train_data = df[train_mask].copy()
            test_data = df[test_mask].copy()

            if len(train_data) == 0 or len(test_data) == 0:
                continue

            self.fit(train_data)  # Fit the model
            y_pred = self.predict(test_data)  # Make predictions
            # print(f"🔍 y_pred type: {type(y_pred)}")
            # print(f"🔍 test_data['E_future']: {type(test_data['E_future'])}")

            test_data[f'E_pred_{self.name}'] = y_pred
            # test_data[f'abs_error_{self.name}'] = np.abs(test_data['E_future'] - y_pred)
            if hasattr(test_data['E_future'], 'to_pandas'):
                # if cuDF.Series，then use .to_pandas()
                test_data[f'abs_error_{self.name}'] = np.abs(test_data['E_future'].to_pandas() - y_pred)
            else:
                # if already pandas.Series，compute directly
                test_data[f'abs_error_{self.name}'] = np.abs(test_data['E_future'] - y_pred)

            test_data[f'abs_error_{self.name}_scaled'] = (
                test_data[f'abs_error_{self.name}'] / test_data['mkt_cap']
            )

            test_data['forecast_year'] = t
            all_results.append(test_data)

        df_forecasts = pd.concat(all_results, axis=0).reset_index(drop=True)
        # turn all cuDF.DataFrame to pandas.DataFrame
        #df_forecasts = pd.concat([df.to_pandas() if isinstance(df, cudf.DataFrame) else df for df in all_results], axis=0).reset_index(drop=True)

        return df_forecasts