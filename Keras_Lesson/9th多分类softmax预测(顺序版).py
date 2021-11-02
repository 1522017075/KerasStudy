import pandas as pd
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import matplotlib
matplotlib.use('TkAgg')

# 初始化数据集
# 花蕊信息数据集(Sepal花蕊长宽 Petal花瓣长宽 Species种类)
data = pd.read_csv('../sources/9thDatas.csv')

# 对种类进行顺序编码(把分类的字符串值改成数字0123)
spc_dic = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
data['Species'] = data.Species.map(spc_dic)

# 获取x(Sepal.Length  Sepal.Width  Petal.Length  Petal.Width)
x = data[data.columns[1:5]]
# 获取y
y = data.Species
print(x.shape, y.shape)
# 训练模型
model = keras.Sequential()
# 输出3个(虽然Species只有一列, 但是仍然输出三个!), 输入4个, 激活函数softmax解决多分类问题
model.add(layers.Dense(3, input_dim=4, activation='softmax'))
# 目标函数使用顺序编码的话, 用sparse_categorical_crossentropy作损失函数计算softmax交叉熵
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)
history = model.fit(x, y, epochs=50)
plt.plot(range(50), history.history.get('acc'))
plt.show()
plt.legend()
