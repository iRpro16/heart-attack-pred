from src.pipeline.utils import load_pkl
from keras.models import load_model
from keras.optimizers import Adam
import tensorflow as tf
import keras
import numpy as np

# File path / load models
scaler_file_path = 'models/scaler.pkl'
new_model = load_model('models/my_model.keras')
new_model.load_weights('models/my_model.weights.h5')

# Scaler
scaler = load_pkl(scaler_file_path)

array = np.array([[69,0,140,239,0,1,151,0,1.8,2,2,2,0.8]])
array_scaled = scaler.transform(array)

prediction = new_model.predict(array_scaled)
pred_max = np.argmax(prediction, axis=1)

print(prediction)
print(pred_max)


