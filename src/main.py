from src.data.make_dataset import Dataset
from src.features.data_processor import Preprocess

if __name__ == "__main__":
    # Get Data
    data = Dataset('dataset/heart.csv')
    
    # Split into X and y
    X, y = data.split_xy()
    
    # Train, test split
    X_train, X_test, y_train, y_test = data.train_test(data.df[X], y)
    
    # Scale data
    scale = Preprocess()
    X_train_scaled, X_test_scaled = scale.scale_data(X_train, X_test)