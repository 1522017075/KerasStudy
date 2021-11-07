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
from keras import regularizers
# kernel_regularizer= 使用它进行Dense层的l2正则, 并输入loss惩罚权重0.005(手动设置)
model.add(layers.Dense(128, kernel_regularizer=regularizers.l1(0.005), input_dim=15, activation='relu'))
model.add(layers.Dense(128, kernel_regularizer=regularizers.l1(0.005), activation='relu'))
model.add(layers.Dense(128, kernel_regularizer=regularizers.l1(0.005), activation='relu'))
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

history = model.fit(x_train, y_train, epochs=1000, validation_data=(x_test, y_test))
print("训练数据集表现:")
model.evaluate(x_train, y_train)
print("测试数据集表现:")
# l1正则loss惩罚权重0.005: loss: 0.4620 - acc: 0.8598
# l1正则loss惩罚权重0.001: loss: 0.9559 - acc: 0.7744
# l2正则loss惩罚权重0.005: loss: 0.7649 - acc: 0.6951
# l2正则loss惩罚权重0.001: loss: 1.1561 - acc: 0.7378
model.evaluate(x_test, y_test)

plt.plot(history.epoch, history.history.get('val_acc'), c='r', label='val_acc')
plt.plot(history.epoch, history.history.get('acc'), c='b', label='acc')
plt.show()
plt.legend()
