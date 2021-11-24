import numpy as np
import keras
from keras import layers
import keras.datasets.mnist as mnist
import tools

(train_image, train_label), (test_image, test_label) = mnist.load_data()
'''
图像数据的shape
    hight width channel
      长    宽   RGB频道(黑白写1或不写, 彩图写3)
conv2d图片输入的形状: (batch, height, width, channels)
train_image.shape: (60000, 28, 28)
需要扩充手写数据集的维度, 用np.expand_dims() -1代表最后一行
'''
train_image = np.expand_dims(train_image, axis=-1)
test_image = np.expand_dims(test_image, axis=-1)

# 开始炼丹
model = keras.Sequential()
# 添加卷积层, 训练64个卷积核, 卷积核大小为3*3, 激活函数为relu, 输入数据的形状(不用写batch, height, width, channels)
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# 添加池化层, 默认为 2*2大小
model.add(layers.MaxPooling2D())
# 扁平化图片, 然后可以接入全连接层
model.add(layers.Flatten())
# 添加全连接层
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))
'''
最后神经网络长这个样子:
    Layer (type)                 Output Shape              Param #   
    conv2d_1 (Conv2D)            (None, 26, 26, 64)        640       
    conv2d_2 (Conv2D)            (None, 24, 24, 64)        36928     
    max_pooling2d_1 (MaxPooling2 (None, 12, 12, 64)        0         
    flatten_1 (Flatten)          (None, 9216)              0         
    dense_1 (Dense)              (None, 256)               2359552   
    dropout_1 (Dropout)          (None, 256)               0         
    dense_2 (Dense)              (None, 10)                2570      
    =================================================================
    Total params: 2,399,690
    Trainable params: 2,399,690
'''
# 多分类问题的loss函数使用sparse_categorical_crossentropy
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)
history = model.fit(train_image, train_label, epochs=5, batch_size=512, validation_data=(test_image, test_label))
# my_model_json = model.to_json() 这种方式可以得到json字符串, 里面只保存了模型的结构, 没有权重(没有训练过)
# model.save_weights('../model/31th_weights.h5) 这种方式可以保存权重到h5文件, 没有模型结构
model.save('../model/31th.h5')
tools.show_accuracy(history)

