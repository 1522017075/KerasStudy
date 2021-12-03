import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import layers
import tools

data = keras.datasets.imdb
max_word = 10000

# 加载电影评论数据集
(x_train, y_train), (x_test, y_test) = data.load_data(num_words=max_word)

# 每条评论的长度都不一样
# print([len(seq) for seq in x_train])

maxlen = 200
# 使用 keras 的方法, 填充评论的长度为 maxlen
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

# 开始炼丹
model = keras.Sequential()
# 词嵌入层 参数介绍(输入维度/单词个数10000, 输出长度/映射到长度为 20 的向量, 输入序列的长度/评论长度为 maxlen)
model.add(layers.Embedding(10000, 20, input_length=maxlen))
# 输出为 128 隐藏单元
model.add(layers.LSTM(128))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['acc']
)

history = model.fit(x_train, y_train, epochs=10, batch_size=256, validation_data=(x_test, y_test))

tools.show_loss(history)
tools.show_accuracy(history)
