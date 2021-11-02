import tensorflow.keras as keras
# 科学计算
import numpy as np
# 画图
import matplotlib.pyplot as plt
# 层
from tensorflow.keras import layers

import tensorflow.keras.datasets.mnist as mnist

(train_image, train_label), (test_image, test_label) = mnist.load_data()

model = keras.Sequential()
# Flatten层可以展平二维数据 (60000, 28, 28) ---> (60000, 28*28)
model.add(layers.Flatten())
# 全连接层, 输出使用64单元的隐藏层, 激活函数relu
model.add(layers.Dense(64, activation='relu'))
# 输出层, 输出使用10单元, 激活函数为softmax
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

# 训练模型(一个批次512张图, 训练50次)
model.fit(train_image, train_label, epochs=50, batch_size=512)

# 测试模型 loss: 0.3546 - acc: 0.9458
model.evaluate(test_image, test_label)
