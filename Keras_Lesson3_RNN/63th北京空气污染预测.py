import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import re
from tensorflow.keras import layers

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

# 数据中 PM2.5 会有 NaN 值, 需要处理之

# print(data['pm2.5'].isna().sum()) 为 2067 条 NaN 数据, 处理思路为就近填充, 把昨天的数据填给 NaN (前 24 条也为 NaN, 直接删掉了)
data = data.iloc[24:].fillna(method='ffill')
