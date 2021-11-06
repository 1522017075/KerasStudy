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

print(x.shape, y.shape)

model = keras.Sequential()
model.add(layers.Dense(32, input_dim=11, activation='relu'))
# 第二层开始不需要输入“输入维度”
model.add(layers.Dense(32, activation='relu'))
# 最后一层使用sigmoid函数， 输出概率值（是否活着）
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['acc']
)

model.fit(x, y, epochs=300)
