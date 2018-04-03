# -*- coding: utf-8 -*-
# @Time     :2018/4/2 下午1:19
# @Author   :李二狗
# @Site     :
# @File     :scikit_learn2.py
# @Software :PyCharm

from sklearn.linear_model import  LinearRegression
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import matplotlib as mpl
import time
import  matplotlib.pyplot as plt

## 设置字符集，防止中文乱码
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

# 数据收集
path1 = 'datas/household_power_consumption_1000.txt'
df = pd.read_csv(path1, sep=';', low_memory=False)
# print(type(df))
# print(df.index)
# print(df.columns)
# print(df.head(5))

# print(df.info())

## 异常数据的处理
new_df = df.replace('?', np.nan)
datas = new_df.dropna(axis=0, how='any')
# print(datas.index)

# print(datas.describe().T)


# 时间与功率的关系
Y = datas['Global_active_power']
print(type(Y))

# 特征提取
def data_format(dt):
    t = time.strptime(' '.join(dt), '%d/%m/%Y %H:%M:%S')
    return (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)

X = datas.iloc[:, 0:2]
X = X.apply(lambda x: pd.Series(data_format(x)), axis=1)
# print(X.head(4))
# print(type(X))


# 对数据进行测试机和数据集的划分
# X： 特征矩阵
# Y：标签或者目标属性
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# print(X_train.describe().T)

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

print(type(X_train))
# print(pd.DataFrame(X_train).describe().T)

lr = LinearRegression(fit_intercept=True)
lr.fit(X_train, Y_train)
Y_predict = lr.predict(X_test)

print('训练集上R^2：', lr.score(X_train, Y_train))
print('测试集上R^2：', lr.score(X_test, Y_test))

mse = np.average((Y_predict-Y_test)**2)
rmse = np.sqrt(mse)
print('rmse:', rmse)

print('模型训练后的系数：', end='')
print(lr.coef_)

print('截距：', lr.intercept_)

# 做图比较
t = np.arange(len(X_test))
plt.figure(facecolor='w')
plt.plot(t, Y_test, 'r-', linewidth=2, label='真实值')
plt.plot(t, Y_predict, 'g-', linewidth=2, label='预测值')

plt.legend(loc='upper left')
plt.title('线性回归预测时间与功率之间的关系', fontsize=20)
plt.grid(b=True)
plt.show()



