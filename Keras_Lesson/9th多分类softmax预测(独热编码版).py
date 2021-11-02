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

# 对种类进行独热编码(学习到pd.get_dummies()函数)后加入data, 删掉原来的种类
data = data.join(pd.get_dummies(data.Species))
del data['Species']
# 对data进行乱序, 因为之前的数据集分好类了不适合训练
index = np.random.permutation(len(data))
data = data.iloc[index]

# 获取x(Sepal.Length  Sepal.Width  Petal.Length  Petal.Width)
x = data[data.columns[1:5]]
# 获取y(setosa  versicolor  virginica)
y = data.iloc[:, -3:]

# 训练模型
model = keras.Sequential()
# 输出3个, 输入4个, 激活函数softmax解决多分类问题
model.add(layers.Dense(3, input_dim=4, activation='softmax'))
# 目标函数使用独热编码的话, 用categorical_crossentropy作损失函数计算softmax交叉熵
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['acc']
)
history = model.fit(x, y, epochs=50)
plt.plot(range(50), history.history.get('acc'))
plt.show()
plt.legend()
