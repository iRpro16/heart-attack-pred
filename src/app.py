from src.pipeline.utils import load_pkl
import numpy as np

# File path / load models
model_file_path = 'models/model.pkl'

# Model
model = load_pkl(model_file_path)

array = np.array([[42,1,136,315,0,1,125,1,1.8,1,0,1,0.12]])

pred = model.predict(array)
pred_prob = model.predict_proba(array)

print(pred_prob)
print(pred)