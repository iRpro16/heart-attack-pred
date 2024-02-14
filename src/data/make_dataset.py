from sklearn.model_selection import train_test_split
from src.logger import logging
import pandas as pd

class Dataset:
    def __init__(self, csv_file):
        logging.info("Read csv file to convert to dataframe")
        self.df = pd.read_csv(csv_file)
        
    # Split X and y to be used for preprocessing
    def split_xy(self):
        logging.info("Split dataset into X and y")
        self.X = self.df.columns.difference(['cp'])
        self.y = self.df['cp']
        return self.X, self.y
    
    def train_test(self, X, y):
        logging.info("Split into train and test")
        return train_test_split(X, y, test_size=0.1, random_state=42)