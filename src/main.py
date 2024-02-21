from src.data.make_dataset import Dataset
from src.model.train_model import Model
from src.pipeline.utils import evaluate_model, print_metrics
from sklearn.utils.class_weight import compute_sample_weight

if __name__ == "__main__":
    # Get Data
    data = Dataset('dataset/heart.csv')
    
    # Split into X and y
    X, y = data.split_xy()
    
    # Train, test split
    X_train, X_test, y_train, y_test = data.train_test(data.df[X], y)
    
    # Params 
    evals = [(X_train, y_train), (X_test, y_test)]
    sample_weights = compute_sample_weight(
        class_weight='balanced',
        y = y_train
    )

    # Get model
    model = Model()
    model.fit_model(X_train, y_train, evals, sample_weights)
    predictions, results = model.predict_model(X_test)
    
    # Store metrics
    ac_score, pr_score = evaluate_model(y_test, predictions)
    
    # Print metrics
    print_metrics(ac_score, pr_score)
