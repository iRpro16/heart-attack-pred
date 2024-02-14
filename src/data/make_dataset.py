import pandas as pd

class Dataset:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        
    # Split X and y to be used for preprocessing
    def split_xy(self):
        self.X = self.df.columns.difference(['cp'])
        self.y = self.df['cp']
        return self.X, self.y