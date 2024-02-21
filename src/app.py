from src.pipeline.utils import load_pkl
from keras.models import load_model
from keras.optimizers import Adam
import tensorflow as tf
import keras
import numpy as np
# File path / load models
scaler_file_path = 'models/scaler.pkl'
model_file_path = 'models/model.pkl'

# Scaler
scaler = load_pkl(scaler_file_path)

# Model
model = load_pkl(model_file_path)

array = [[43,0,132,341,1,0,136,1,3.0,1,0,3,0.21]]

pred = model.predict(array)

print(pred)