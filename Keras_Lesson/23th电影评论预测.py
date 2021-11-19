import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras import layers

data = keras.datasets.imdb

# 引入一万个单词的映射关系 (之后电影评论的单词会被替换为数字, 相当于一个字典)
max_word = 10000

(x_train, y_train), (x_test, y_test) = data.load_data(num_words=max_word)

# 获得"单词":"序号" 键值对
word_index = data.get_word_index()
# 获得"序号":"单词" 键值对
index_word = dict((value, key) for key, value in word_index.items())


# tips:
# 独热编码(one-hot) : 诸多列中只符合一列
# k热编码(k-hot) : 诸多列中可以符合k列
# 文本的向量化(传统机器学习: tf idf) 今天用k-hot
def k_hot(seqs, dim=10000):
    # seqs(多少条评论)个维度, 每一维有10000个0的向量, 哪个单词亮了给哪个坑里填, 最后得到一个k-hot的文本向量化结果
    result = np.zeros((len(seqs), dim))
    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
    # i是第几条评论, seq是具体的评论内容
    for i, seq in enumerate(seqs):
        result[i, seq] = 1
    return result


# x_train.shape 为 (25000, 10000) 2w5k条, 1w个列的向量
x_train = k_hot(x_train)
x_test = k_hot(x_test)

# 开始炼丹
model = keras.Sequential()
# 输出32个隐藏单元, 输入维度x_train(25000, 10000)不需要关心25000条评论, 而是看他有多少列(10000)
model.add(layers.Dense(16, input_dim=10000, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['acc']
)
history = model.fit(x_train, y_train, epochs=15, batch_size=256, validation_data=(x_test, y_test))

# 其实根据图像来看, 有很大的过拟合, 可以采用降低layers.Dense(16)适当减少隐藏单元, 或者加入DropOut层优化
plt.plot(history.epoch, history.history.get('loss'), c='r', label='loss')
plt.plot(history.epoch, history.history.get('val_loss'), c='b', label='val_loss')
plt.legend()
plt.show()
