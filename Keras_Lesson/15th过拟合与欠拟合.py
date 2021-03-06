import pandas as pd
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import matplotlib

matplotlib.use('TkAgg')

# 德国信用卡欺诈（15列特征， 最后二分类结果）header=None表示数据文件没有表头
data = pd.read_csv('../sources/15thDatas.csv', header=None)

# 用numpy array做输入
x = data.iloc[:, :-1].values
# 把y(replace成只有0和1)变成n个一位的数, 用reshape函数
y = data.iloc[:, -1].replace(-1, 0).values.reshape(-1, 1)

model = keras.Sequential()
# 用input_shape就得是上面这个格式(TensorFlow 1.15)  TensorFlow2.2就可以写成input_shape=(None, 15)
# model.add(layers.Dense(128, input_shape=(x.shape[1],), activation='relu'))
# 把第二层输入改成1，就可以造成欠拟合
model.add(layers.Dense(128, input_dim=15, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

history = model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['acc']
)

# 过拟合的话: loss: 0.0633 - acc: 0.9893
# model.fit(x, y, epochs=1000)

#_____________________分割线_____________________________

# 75%做训练 25%做测试
x_train = x[:int(len(x) * 0.75)]
x_test = x[int(len(x) * 0.75):]
y_train = y[:int(len(y) * 0.75)]
y_test = y[int(len(y) * 0.75):]

history = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))
# 拟合出来的: loss: 0.0853 - acc: 0.9796
print("训练数据集表现:")
model.evaluate(x_train, y_train)
# 测试出来的: loss: 2.6663 - acc: 0.7256
print("测试数据集表现:")
model.evaluate(x_test, y_test)

plt.plot(history.epoch, history.history.get('val_acc'), c='r', label='val_acc')
plt.plot(history.epoch, history.history.get('acc'), c='b', label='acc')
plt.show()
plt.legend()
