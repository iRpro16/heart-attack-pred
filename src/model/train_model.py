from src.pipeline.utils import save_object
from src.logger import logging
from xgboost import XGBClassifier
import os

class Model():
    logging.info("Create new model")
    def __init__(self):
        self.model_trainer_config = os.path.join("models", "model.pkl")
        
    def fit_model(self, x_train, y_train, evals, sample_weights):
        self.model = XGBClassifier(
            learning_rate = 0.1,
            max_depth = 10, 
            n_estimators = 200,
            min_child_weight = 6,
            eval_metric = 'mlogloss'
        )
        # Fit model with evals and sample_weights
        self.model.fit(x_train, y_train, eval_set=evals, sample_weight = sample_weights)
        
        # Save model
        logging.info("Save model")
        save_object(
            obj=self.model,
            file_path=self.model_trainer_config
        )
    
    logging.info("Create prediction method")
    # Predict model
    def predict_model(self, x_test):
        self.prediction = self.model.predict(x_test)
        self.results = self.model.evals_result()
        
        return self.prediction, self.results
        