from .base_model import BaseModel

class RandomWalkModel(BaseModel):
    """Random Walk forecasting model."""

    def __init__(self):
        super().__init__("rw")

    def fit(self, train_data):
        # Random Walk doesn't require fitting
        pass

    def predict(self, test_data):
        return test_data['E'].values