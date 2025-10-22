from .base_model import BaseModel
from sklearn.linear_model import LinearRegression

class ARModel(BaseModel):
    """Autoregressive forecasting model."""

    def __init__(self):
        super().__init__("ar")
        self.model = LinearRegression(fit_intercept=False)

    def fit(self, train_data):
        """Fits an AR(1) model: E_t+1 = beta * E_t + error"""
        X_train = train_data[['E']]
        y_train = train_data['E_future']

        self.model.fit(X_train, y_train)

    def predict(self, test_data):
        """Generates predictions for the test data using the fitted AR model."""
        X_test = test_data[['E']]
        y_pred = self.model.predict(X_test)

        return y_pred
