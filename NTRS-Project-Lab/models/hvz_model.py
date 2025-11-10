import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from models.base_model import BaseModel
from sklearn.linear_model import LinearRegression
from models.utils import winsorize_series


class HVZModel(BaseModel):
    def __init__(self):
        super().__init__("hvz")
        self.input_list = ["at", "dvc", "DDi", "E", "NegEi", "AC"]
        self.model = LinearRegression()

    def fit(self, train_data):
        train_data["DDi"] = (train_data["dvc"] > 0).astype(int)
        train_data["NegEi"] = (train_data["E"] < 0).astype(int)
        train_data["AC"] = train_data["E"] - train_data["oancf"]

        # Scale the predictors and target
        X_train = train_data[self.input_list]
        y_train = train_data["E_future"]

        self.model.fit(X_train, y_train)

    def predict(self, test_data):
        test_data["DDi"] = (test_data["dvc"] > 0).astype(int)
        test_data["NegEi"] = (test_data["E"] < 0).astype(int)
        test_data["AC"] = test_data["E"] - test_data["oancf"]

        X_test = test_data[self.input_list]
        y_pred = self.model.predict(X_test)
        return y_pred