import pandas as pd
from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        
    # Split X and y to be used for preprocessing
    def split_xy(self):
        self.X = self.df.columns.difference(['cp'])
        self.y = self.df['cp']
        return self.X, self.y
    
    def train_test(self, X, y):
        return train_test_split(X, y, test_size=0.1, random_state=42)