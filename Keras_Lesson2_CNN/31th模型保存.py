import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras import layers
from keras.models import load_model
import keras.datasets.mnist as mnist

(train_image, train_label), (test_image, test_label) = mnist.load_data()
train_image = np.expand_dims(train_image, axis=-1)
test_image = np.expand_dims(test_image, axis=-1)

my_model = load_model('../model/31th.h5')

print(my_model.evaluate(test_image, test_label))
