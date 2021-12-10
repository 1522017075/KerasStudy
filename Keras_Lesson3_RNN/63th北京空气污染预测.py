import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import re
from tensorflow.keras import layers

import tools

"""
    目标: 根据多种数据预测 PM2.5 的值
"""
data = pd.read_csv('../sources/63thDatas_PRSA_data_2010.1.1-2014.12.31.csv')
"""
print(data.head()) 数据长这样, 时间 DEWP: 露点, TEMP: 温度, PRES: 风向, Iws: 风速, Is: 累积雪量, Ir: 累积雨量
   No  year  month  day  hour  pm2.5  DEWP  TEMP    PRES cbwd    Iws  Is  Ir
0   1  2010      1    1     0    NaN   -21 -11.0  1021.0   NW   1.79   0   0
1   2  2010      1    1     1    NaN   -21 -12.0  1020.0   NW   4.92   0   0
2   3  2010      1    1     2    NaN   -21 -11.0  1019.0   NW   6.71   0   0
3   4  2010      1    1     3    NaN   -21 -14.0  1019.0   NW   9.84   0   0
4   5  2010      1    1     4    NaN   -20 -12.0  1018.0   NW  12.97   0   0
"""

# 数据预处理 1: 数据中 PM2.5 会有 NaN 值, 需要处理之
# print(data['pm2.5'].isna().sum()) 为 2067 条 NaN 数据, 处理思路为就近填充, 把昨天的数据填给 NaN (前 24 条也为 NaN, 直接删掉了)
data = data.iloc[24:].fillna(method='ffill')

# 数据预处理 2: 把多列时间(年月日小时) 合并成一列
import datetime

# 利用库函数 datetime, 创建 lambda 函数, axis=1: data 数据的每一行为一个输入
data['tm'] = data.apply(lambda x: datetime.datetime(year=x['year'],
                                                    month=x['month'],
                                                    day=x['day'],
                                                    hour=x['hour']), axis=1)
# 去掉多余的列
data.drop(columns=['year', 'month', 'day', 'hour', 'No'], inplace=True)
# 将时间列作为序号列
data = data.set_index('tm')

# 数据预处理 3: 把风向独热编码化(data.cbwd.unique() => ['SE' 'cv' 'NW' 'NE'])
data = data.join(pd.get_dummies(data.cbwd))
del data['cbwd']

"""
    最终处理好的数据:
                     pm2.5  DEWP  TEMP    PRES   Iws  Is  Ir  NE  NW  SE  cv
tm                                                                          
2010-01-02 00:00:00  129.0   -16  -4.0  1020.0  1.79   0   0   0   0   1   0
2010-01-02 01:00:00  148.0   -15  -4.0  1020.0  2.68   0   0   0   0   1   0
2010-01-02 02:00:00  159.0   -11  -5.0  1021.0  3.57   0   0   0   0   1   0
2010-01-02 03:00:00  181.0    -7  -5.0  1022.0  5.36   1   0   0   0   1   0
2010-01-02 04:00:00  138.0    -7  -5.0  1022.0  6.25   2   0   0   0   1   0

    训练策略:
        以当前点为基准, 向前 5*24 条数据为训练数据, 向后第 24 条数据(一天之后的此刻)为预测数据, 进行训练
        每一个时刻的预测都会涉及到 5*24 + 1 这么多条数据
    策略缺陷: (TODO: 第几节课会改进)
        作为训练数据的重复次数达到 5*24 次, 会有大量重复数据占用内存
"""
seq_length = 5*24
delay = 24

# 从源数据采样训练数据, 采集到倒数第七天.(每 5*24 + 24 条数据作为一整条数据放入到 data_[]数组中)
data_ = []
for i in range(len(data) - seq_length - delay):
    data_.append(data.iloc[i: i + seq_length + delay])

data_ = np.array([df.values for df in data_])
"""
print(data_[0].shape) 
(144, 11) 每一条数据有 144 行, 11 列
print(data_.shape) 
(43656, 144, 11) 一共有 43656 条这种数据
"""

# 乱序处理 data_
np.random.shuffle(data_)

# 提取训练数据x: 每一条大数据的, 前 120 行, 的每一列
# 提取训练数据y: 每一条大数据的, 最后一行, 的第 0 列(PM2.5 的值)
x = data_[:, :5*24, :]
y = data_[:, -1, 0]

# 切割 80%为训练数据, 20%为测试数据
split_b = int(data_.shape[0] * 0.8)
train_x = x[:split_b]
train_y = y[:split_b]
test_x = x[split_b:]
test_y = y[split_b:]

# 数据标准化 计算均值mean和方差std axis=0: 列为单位(暂时不明白), 然后对数据减均值除方差即可. (无须对预测结果做标准化)
mean = train_x.mean(axis=0)
std = train_x.std(axis=0)

train_x = (train_x - mean)/std
test_x = (test_x - mean)/std

"""
# 用全连接炼丹
batch_size = 128

model = keras.Sequential()

# train_x.shape 为(34924, 120, 11), 不能直接被输入, 需要 Flatten 展开数据为一维
# (展开之后其实也就是失去了[时间]维度, 所以效果会比 LSTM 差劲)
model.add(layers.Flatten(input_shape=(train_x.shape[1:])))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))

# mse: Mean Square Error 均方误差
# mae: mean_absolute_error 平均绝对误差
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

history = model.fit(train_x, train_y, batch_size=batch_size, epochs=50, validation_data=(test_x, test_y))

tools.show_loss(history)
最终平均误差大概是 50 左右, 即, 正确值为 300, 那么预测数据可能是 250 ~ 350 之间
"""

# LSTM 炼丹: 对时序数据
batch_size = 128
model = keras.Sequential()
# train_x.shape 为(34924, 120, 11), 可以直接输入 LSTM, 32 个隐藏单元
model.add(layers.LSTM(32, input_shape=(120, 11)))
model.add(layers.Dense(1))

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

history = model.fit(train_x, train_y, batch_size=batch_size, epochs=50, validation_data=(test_x, test_y))

tools.show_loss(history)
