import tensorflow.keras as keras
# 科学计算
import numpy as np
# 画图
import matplotlib.pyplot as plt
# 层
from tensorflow.keras import layers

import tensorflow.keras.datasets.mnist as mnist

(train_image, train_label), (test_image, test_label) = mnist.load_data()
train_image = train_image / 256
# 优化模型步骤1: 增大网络容量, 直到过拟合
model = keras.Sequential()
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
# 优化模型步骤2: 采取措施抑制过拟合
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
# 优化模型步骤2: 采取措施抑制过拟合
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

# 编译模型
#   优化函数: adam
#   损失函数: sparse_categorical_crossentropy
#   评价函数: 用于评估当前训练模型的性能
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)

# 这里训练由50次改为200次, 因为DropOut了一部分数据, 所以要多训练一些
# 训练模型(一个批次512张图, 训练200次) validation_data查看训练过程中测试数据的表现
model.fit(train_image, train_label, epochs=200, batch_size=512, validation_data=(test_image, test_label))

# 增大网络容量后DropOut的测试模型 - acc: 0.9733
model.evaluate(test_image, test_label)

