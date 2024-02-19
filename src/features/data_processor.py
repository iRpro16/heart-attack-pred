from sklearn.preprocessing import StandardScaler
from src.pipeline.utils import save_object
from src.logger import logging
import os

class Preprocess:
    def __init__(self):
        # Path to save scaler
        self.scaler_config = os.path.join('models','scaler.pkl') 
    
    def scale_data(self, X_train, X_test):
        logging.info("Scaled the data")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.fit_transform(X_test)
        
        # Save scaler
        logging.info("Save scaler")
        save_object(
            obj = self.scaler,
            file_path=self.scaler_config
        )
      
        
        return X_train_scaled, X_test_scaled