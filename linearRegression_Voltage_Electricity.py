# -*- coding: utf-8 -*-
# @Time     :2018/4/3 下午1:11
# @Author   :李二狗
# @Site     :
# @File     :linearRegression_Voltage_Electricity.py
# @Software :PyCharm

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

# 1、加载数据
# 日期、时间、有功功率、无功功率、电压、电流、厨房用电功率、洗衣服用电功率、热水器用电功率
path1 = 'datas/household_power_consumption_1000.txt'
df = pd.read_csv(path1, sep=';', low_memory=True)

# 2、异常数据处理
# 替换非法字符为np.nan
new_df = df.replace('?', np.nan)
# 只要有一个数据为空，就进行行删除操作
datas = new_df.dropna(axis=0, how='any')

# 3、特征选择
X2 = datas.iloc[:, 2:4]
Y2 = datas.iloc[:, 5]

# 4、数据分割
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2, test_size=0.2, random_state=0)

# 5、数据归一
scaler2 = StandardScaler()
X2_train = scaler2.fit_transform(X2_train) # 训练并转换
X2_test = scaler2.transform(X2_test) ## 直接使用在模型构建数据上进行一个数据标准化操作

# 6、模型训练
lr2 = LinearRegression()
lr2.fit(X2_train, Y2_train)

# 结果预测
Y2_predict = lr2.predict(X2_test)

## 模型评估
print("电流预测准确率: ", lr2.score(X2_test, Y2_test))
print("电流参数:", lr2.coef_)

t = np.arange(len(X2_test))
plt.figure(facecolor='w')
plt.plot(t, Y2_test, 'r-', linewidth=2, label='真实值')
plt.plot(t, Y2_predict, 'g-', linewidth=2, label='预测值')
plt.legend(loc='lower right')
plt.title('线性回归预测功率与电流之间的关系', fontsize=20)
plt.grid(b=True)
plt.show()




