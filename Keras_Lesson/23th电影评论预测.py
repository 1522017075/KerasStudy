import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras import layers

data = keras.datasets.imdb

# 引入一万个单词的映射关系 (之后电影评论的单词会被替换为数字, 相当于一个字典)
max_word = 10000

(x_train, y_train), (x_test, y_test) = data.load_data(num_words=max_word)

print(x_train.shape)
