import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import Adam
from keras import Model
from keras.regularizers import l2
from keras.layers import Dense
from src.logger import logging

class MyModel():
    def __init__(self):
        self.model = Sequential([
            Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
            Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
            Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
            Dense(12, activation='relu', kernel_regularizer=l2(0.001)),
            Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
            Dense(4, activation='softmax')
        ])
        
    # Compile model
    def compile_model(self):
        self.model.compile(
            optimizer=Adam(0.001),
            loss='sparse_categorical_crossentropy'
        )
        
    # Fit model
    def fit_model(self, x_train, y_train, x_test, y_test):
        self.history = self.model.fit(
            x_train, y_train,
            validation_data=(x_test, y_test),
            epochs=50, 
            verbose=0
        )
        
        # Save model and weights
        logging.info("Save model and weights")
        self.model.save('models/my_model.keras')
        self.model.save_weights('models/my_model_weights.h5')
        
        return self.history
        
    # Predict method
    def predict_model(self, x_test):
        self.predict = self.model.predict(x_test)
        return self.predict
    
logging.info("Created new model")