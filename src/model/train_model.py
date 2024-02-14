import tensorflow as tf
from keras.models import Sequential
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.layers import Dense
from src.logger import logging

class MyModel():
    
    def __init__(self):
        pass
    
    # Get model using Sequential API
    def get_model(self):
        logging.info("Initialize model")
        self.model = Sequential(
            [
                Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
                Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
                Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
                Dense(12, activation='relu', kernel_regularizer=l2(0.001)),
                Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
                Dense(4, activation='softmax', kernel_regularizer=l2(0.001)),
            ]
        )
        self.model.compile(
            optimizer=Adam(0.001),
            loss = 'sparse_categorical_crossentropy'
        )
    
    # Fit model to training and testing set
    def fit_model(self, x_train, y_train, x_test, y_test):
        logging.info("Fit model")
        self.model.fit(
            x_train, y_train,
            validation_data=(x_test, y_test),
            epochs=50,
            verbose=0
        )
        
        # Get prediction
        self.predict = self.model.predict(x_test)
        return self.predict