from sklearn.preprocessing import StandardScaler
import os

class Preprocess:
    def __init__(self):
        #self.scaler_config = os.path.join('models','scaler.pkl')
        pass 
    
    def scale_data(self, X_train, X_test):
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.fit_transform(X_test)
        return X_train_scaled, X_test_scaled