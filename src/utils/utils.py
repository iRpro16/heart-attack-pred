from sklearn.metrics import accuracy_score, precision_score
import pickle
import os

# Evaluate model accuracy
def evaluate_model(true, predicted):
    ac_score = accuracy_score(true, predicted)
    pr_score = precision_score(true, predicted, average='micro')
    return ac_score, pr_score

# Save object
def save_object(file_path, obj):
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    with open(file_path, 'wb') as file_obj:
        pickle.dump(obj, file_obj)
        
# Load pickle
def load_pkl(pickle_file_path):
    with open(pickle_file_path, 'rb') as f:
        obj = pickle.load(f)
        return obj

