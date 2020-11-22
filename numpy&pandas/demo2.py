# -*- codeing = utf-8 -*-
# @Time : 2020/11/21 15:49
# @Author : 王浩
# @File : demo2.py
# @Software : PyCharm
import pandas as pd
import numpy as np

s1 = pd.Series([1, 3, np.nan, pd.Series([1])])
s2 = pd.Series([1, 3, np.nan, 2])
print(s1)
print(s2)

dates = pd.date_range('20201121', periods=5)
print(dates)

df1 = pd.DataFrame(np.random.randn(5, 4), index=dates, columns=['a', 'b', 'c', 'd'])
print(df1)
df2 = pd.DataFrame(np.random.randn(3, 4))
print(df2)

# 内部选用字典
df3 = pd.DataFrame({'A': 1.,
                    'B': pd.Timestamp('20201121'),
                    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                    'D': pd.Categorical(["test", "train", "test", "train"]),
                    "E": np.array([3]*4, dtype='int32'),
                    "F": "foo"})
print(df3)
print(df3.dtypes)
# 输出行的序号（index）
print(df3.index)
# 输出列的序号（columns）
print(df3.columns)
# 输出df的值（values）
print(df3.values)

# 描述数学性质
print(df3.describe())

# 转置
print(df3.T)

# 排序，index=1, 对行的索引值进行排序，ascending = False,倒序
print(df3.sort_index(axis=1, ascending=False))

# 对里面的值进行排序
print(df3.sort_values(by='D'))
