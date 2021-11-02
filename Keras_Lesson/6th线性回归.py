import tensorflow.keras as keras
import pandas as pd
from tensorflow.keras import layers

# 广告投入与利润数据集
data = pd.read_csv('../sources/6thDatas.csv')
# print(data.head())
x = data[data.columns[1:-1]]
# iloc取值
y = data.iloc[:, -1];

model = keras.Sequential()
# y_pred = w1 * x1 + w2 * x2 + w3 * x3 + bias
model.add(layers.Dense(1, input_dim=3))

# 回归问题的损失函数: 均方差
model.compile(
    optimizer='adam',
    loss='mse'
)

model.fit(x, y, epochs=2000)

# 因为训练用的是DataFrame
predict_y = model.predict(pd.DataFrame([[300, 0, 0]]))
print(predict_y) # [[16.523697]]
