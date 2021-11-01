# 什么是逻辑回归（解决[分类问题]，如输入图片是不是猫
#   将输入通过Sigmoid函数转换为True/False（其概率是否大于0.5
#   逻辑回归使用的损失函数：交叉熵损失函数（类似log的函数图像
#       交叉熵刻画的是实际输出（概率）与期望输出（概率）的距离，其值越小，两个概率分布就越接近
#   keras中使用binary_crossentropy来计算二元交叉熵
# 对数几率回归解决的是二分类问题，对于[属于多个分类中哪一个的问题]，使用softmax函数
#   如，这个人来自北上广深的哪里，它是对数几率回归在N个可能不同的值上的推广
#   神经网络的原始输出不是一个概率值，实质上知识输入的数值做了复杂加权与非线性处理后的值
#       将这个输出值变为概率分布就是Softmax层的作用
#   softmax要求每个样本必须属于某个类别，且所有可能的样本均会被覆盖
#       softmax各个样本分量之和为1，当只有两个类别时，与对数几率回归完全相同
#   keras中使用
#       categorical_crossentropy & sparse_categorical_crossentropy
#       来计算softmax交叉熵
import pandas as pd
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import matplotlib
matplotlib.use('TkAgg')
# 数据预处理
# 泰坦尼克号获救预测（字符串值要处理成数字 如男女 =》 0/1
#   目标 survived 0/1
#   变量 Pclass船票等级 Sex Age SibSp兄弟姐妹同船 Parch父母同船 Fare船费 Embarked船舱位置
data = pd.read_csv('../sources/7thDatas.csv')

y = data.Survived
x = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

# 使用[独热编码], 增加列 编码船舱位置, 删掉原始Embarked
x.loc[:, 'Embarked_S'] = (x.Embarked == 'S').astype('int')
x.loc[:, 'Embarked_C'] = (x.Embarked == 'C').astype('int')
x.loc[:, 'Embarked_Q'] = (x.Embarked == 'Q').astype('int')
del x['Embarked']

# 把男女编码成1
x['Sex'] = (x.Sex == 'male').astype('int')

# 有些Age为NaN, 使用均值填充
x['Age'] = x.Age.fillna(x.Age.mean())

# 使用[独热编码], 增加列 编码船票等级(因为是一等二等三等, 一种序列特征而非线性关系的数值), 删掉原始Pclass
x.loc[:, 'p1'] = (x.Pclass == 1).astype('int')
x.loc[:, 'p2'] = (x.Pclass == 2).astype('int')
x.loc[:, 'p3'] = (x.Pclass == 3).astype('int')
del x['Pclass']

# 炼丹
model = keras.Sequential()
# y_predict = (w1 * x1 + w2 * x2 + ... + w11 * x11) + b
# 然后对y_predict进行activation激活函数运算处理
model.add(layers.Dense(1, input_dim=11, activation='sigmoid'))
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['acc']
)
# 存下成功率, 画图看看
history = model.fit(x, y, epochs=100)
plt.plot(range(100), history.history.get('acc'))
plt.show()
plt.legend()
