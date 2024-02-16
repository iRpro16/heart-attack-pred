import tensorflow as tf
from keras.models import load_model
import numpy as np

new_model = load_model('models/my_model.keras')

array = np.array([[61,1,140,207,0,0,138,1,1.9,2,1,3,0.13]])
