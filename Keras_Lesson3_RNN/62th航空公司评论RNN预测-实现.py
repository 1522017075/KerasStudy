import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import re
from tensorflow.keras import layers
import tools

data = pd.read_csv('../sources/60thDatas.csv')

# 取出两列, airline_sentiment是否正面评价, text 是内容,
# negative    9178
# neutral     3099
# positive    2363
data = data[['airline_sentiment', 'text']]

# 取出所有的正面评价 与 负面评价
data_p = data[data.airline_sentiment == 'positive']
data_n = data[data.airline_sentiment == 'negative']

# 因负面评价有 9k, 正面 2k, 为了正负一样多, 只拿负面评价的 前2k
data_n = data_n.iloc[:len(data_p)]
# 把 正负评价组合在一起构成数据集
data = pd.concat([data_n, data_p])

# 把正负评价乱序处理
data = data.sample(len(data))

# 把正负评价用 0/1 表示, 然后删掉原来的列
data['review'] = (data.airline_sentiment == 'positive').astype('int')
del data['airline_sentiment']

# 把文本向量化
# 正则表达式, 只保留评论的英文字母和常见标点
token = re.compile('[A-Za-z]+|[!?,.()]')


def reg_text(text):
    new_text = token.findall(text)
    new_text = [word.lower() for word in new_text]
    return new_text


# 把正则应用到数据集中
data['text'] = data.text.apply(reg_text)

# 获取数据集中所有的单词, 用set去重, 构造一个字典, 英文单词与位置的映射, 用作Embeding
word_set = set()
for text in data.text:
    for word in text:
        word_set.add(word)

# 所有单词的个数
max_word = len(word_set) + 1

# 把 set 转为 list, 对下标进行编码
word_list = list(word_set)
# 获取英文单词与位置的映射的字典, (+1是因为不让单词编码成 0, 后续要用 0 填充评论为相同长度)
word_index = dict((word, word_list.index(word) + 1) for word in word_list)

# 把字典应用于评论中, 单词转为数字
data_ok = data.text.apply(lambda x: [word_index.get(word, 0) for word in x])

# 最长的评论长度
maxlen = max([len(i) for i in data_ok])

# 把每一条评论都填充成 maxlen 那么长
data_ok = keras.preprocessing.sequence.pad_sequences(data_ok.values, maxlen=maxlen)

# 开始炼丹
model = keras.Sequential()
# max_word : 总得单词个数, 50: 映射成的密集向量的长度, input_length=maxlen : 每条评论的长度
model.add(layers.Embedding(max_word, 50, input_length=maxlen))
# 64 隐藏单元的 LSTM TODO: (Mac M1 芯片此出需求 numpy 版本 1.19, 但是并无适配, 所以需要去 window 测试一下)
model.add(layers.LSTM(64))
# 二分类 sigmoid 激活函数
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='binary_corssentropy',
    metrics=['acc']
)

# validation_split 把 20% 数据作为测试数据
history = model.fit(data_ok, data.review.values, epochs=10, batch_size=128, validation_split=0.2)

tools.show_loss(history)
tools.show_accuracy(history)
