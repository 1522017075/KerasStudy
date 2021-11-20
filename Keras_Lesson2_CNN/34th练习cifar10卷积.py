import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras import layers

# https://www.cs.toronto.edu/~kriz/cifar.html 数据集的样子 十个种类的彩色图片
cifar = keras.datasets.cifar10
(train_image, train_label), (test_image, test_label) = cifar.load_data()

# 归一化(把rgb的0 ~ 255映射为0 ~ 1的数字)
train_image = train_image/255
test_image = test_image/255

# 练习炼丹
model = keras.Sequential()
# 四个卷积层, 卷完不改变图片大小, 池化一下, 扔一点防止过拟合
model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Dropout(0.25))

# 俩卷积层, 增大卷积核, 不改变图片大小, 池化, drop
model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Dropout(0.25))

# 展平, 全连接, 隐藏单元512个, softmax分类10个结果
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# 目标函数使用顺序编码的话, 用sparse_categorical_crossentropy作损失函数计算softmax交叉熵
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)

model.fit(train_image, train_label, epochs=10, batch_size=512)

# 增大网络容量和卷积核数量, 正确率虽然上去了, 但是可能会过拟合(也有在DropOut), 正确率大概在85%往上
model.evaluate(test_image, test_label)

