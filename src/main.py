from src.data.make_dataset import Dataset
from src.features.data_processor import Preprocess
from src.model.train_model import MyModel
from src.pipeline.utils import evaluate_model, print_metrics
import numpy as np


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
    
    # Get model
    model = MyModel()
    model.get_model()
    prediction = model.fit_model(X_train_scaled, y_train, X_test_scaled, y_test)
    pred_max = np.argmax(prediction, axis=1)
    
    # Store metrics
    ac_score, pr_score = evaluate_model(y_test, pred_max)
    
    # Print metrics
    print_metrics(ac_score, pr_score)