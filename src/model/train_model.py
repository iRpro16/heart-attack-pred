from xgboost import XGBClassifier
from src.pipeline.utils import save_object
import os

class Model():
    def __init__(self):
        self.model_trainer_config = os.path.join("models", "model.pkl")
        
    def fit_model(self, x_train, y_train):
        self.model = XGBClassifier()
        self.model.fit(x_train, y_train)
        
        # Save model
        save_object(
            obj=self.model,
            file_path=self.model_trainer_config
        )
        
        return self.model