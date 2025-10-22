import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from models.base_model import BaseModel

class RIModel(BaseModel):
    """RI Model"""
    def __init__(self):
        super().__init__("ri")
        self.input_list = ["NegEt", "E", "NegE_x_E", "B", "TACC"]
        self.model = LinearRegression()

    def fit(self, train_data: pd.DataFrame):
        df = train_data.copy()
        df = df.sort_values(["gvkey", "year"])

        # 1) Define next-year values for ib and spi to compute future earnings.
        df["ib_future"]  = df.groupby("gvkey")["ib"].shift(-1)
        df["spi_future"] = df.groupby("gvkey")["spi"].shift(-1)

        # 2) Compute per-share earnings and book value.
        df["E"]        = df["ib"] - df["spi"]
        df["E_future"] = df["ib_future"] - df["spi_future"]
        df["B"]        = df["ceq"]

        # 3) Calculate total accruals (TACC) from Richardson et al. (2005)
        df["WC"]  = (df["act"] - df["che"]) - (df["lct"] - df["dlc"])
        df["NCO"] = (df["at"] - df["act"] - df["ivao"]) - (df["lt"] - df["lct"] - df["dltt"])
        df["FIN"] = (df["ivst"] + df["ivao"]) - (df["dltt"] + df["dlc"] + df["pstk"])

        df["dWC"]  = df.groupby("gvkey")["WC"].diff().fillna(0)
        df["dNCO"] = df.groupby("gvkey")["NCO"].diff().fillna(0)
        df["dFIN"] = df.groupby("gvkey")["FIN"].diff().fillna(0)
        df["TACC"] = df["dWC"] + df["dNCO"] + df["dFIN"]

        # 4) Create negative earnings dummy and interaction term.
        df["NegEt"] = (df["E"] < 0).astype(int)
        df["NegE_x_E"] = df["NegEt"] * df["E"]
        
        df = df.dropna(subset=["E_future"])

        X_train = df[self.input_list]
        y_train = df["E_future"]
        self.model.fit(X_train, y_train)

    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        df = test_data.copy()
        df = df.sort_values(["gvkey", "year"])

        df["E"] = df["ib"] - df["spi"]
        df["B"] = df["ceq"]
        df["WC"]  = (df["act"] - df["che"]) - (df["lct"] - df["dlc"])
        df["NCO"] = (df["at"] - df["act"] - df["ivao"]) - (df["lt"] - df["lct"] - df["dltt"])
        df["FIN"] = (df["ivst"] + df["ivao"]) - (df["dltt"] + df["dlc"] + df["pstk"])
        df["dWC"]  = df.groupby("gvkey")["WC"].diff().fillna(0)
        df["dNCO"] = df.groupby("gvkey")["NCO"].diff().fillna(0)
        df["dFIN"] = df.groupby("gvkey")["FIN"].diff().fillna(0)
        df["TACC"] = df["dWC"] + df["dNCO"] + df["dFIN"]
        df["NegEt"] = (df["E"] < 0).astype(int)
        df["NegE_x_E"] = df["NegEt"] * df["E"]

        df = df.dropna(subset=["E_future"])

        X_test = df[self.input_list]
        y_pred = self.model.predict(X_test)
        return y_pred