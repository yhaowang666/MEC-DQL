# -*- codeing = utf-8 -*-
# @Time : 2020/11/21 9:51
# @Author : 王浩
# @File : demo1.py
# @Software : PyCharm

import numpy as np

# np 定义二维数组
'''
array = np.array([[1, 2, 3],
                  [4, 5, 6]])

print(array)
print('number of dim:', array.ndim)  # 维度
print('shape:', array.shape)
print('size:', array.size)
'''

'''
# numpy 定义数组的类型(int, float, float32)
array = np.array([1, 233, 3], dtype=float)
print(array.dtype)
# zeros 定义全为 0 的数组
array = np.zeros((3, 4))
print(array)
# zeros 定义全为 1 的数组
array = np.ones((3, 4))
print(array)
# arange 定义从[a,b)步长为c的数组 a=1, b=10, c=2
array = np.arange(1, 10, 2)
print(array)
array = np.arange(15)  # 0-14
print(array)
array = np.arange(15).reshape(3, 5)  # 重新定义形状
print(array)
# linspace 将[a,b]的数组等分为c分
array = np.linspace(0, 10, 6)
print(array)
print(array.dtype)
array = array.reshape((2, 3))
print(array)
'''

'''
# numpy 数学运算
array1 = np.arange(2, 17, 2)
array2 = np.linspace(1, 15, 8)
print(array1, array2)
print(array1 - array2)
print(array1 + array2)
print(array1**2)  # 注意在 python 中 ** 代表了幂次运算
print(np.sin(array1))  # np 中提供了一些函数运算，如sin， cos
print(np.cos(array1))
array3 = (array1 < 10)
print(array3)

print(array1*array2)  # 矩阵对应位置相乘，维度一致
print(np.sum(array1*array2))
print(np.dot(array1, array2))  # 矩阵乘法， dot有时会智能地纠正矩阵的维度
print(array1.dot(array2))

# 随机生成[0,1]之间的数字，（3，4）为数列形状
array4 = np.random.random((3, 4))
print(array4)
print(np.max(array4))
print(np.min(array4))
print(np.max(array4, axis=0))  # axis=0 在每一列当中求解，输出是一个一维数列
print(np.min(array4, axis=1))  # axis=1 在每一行当中求解，输出是一个一维数列
'''

'''
array = np.arange(0, 16).reshape((4, 4))
print(array)
print(np.argmin(array))  # 0
print(array.argmax())   # 13
print(array.mean())   # 平均值
print(np.average(array))  # 平均值(老版本)
print(np.median(array))  # 中位数
print(np.cumsum(array))   # 累加
print(np.diff(array))   # 累差
print(np.nonzero(array))  # 输出非零元素的行和列，不包含0行0列

array1 = np.random.random((3, 4))
print(array1)
print(np.sort(array1))  # 逐行排序
print(np.sort(array1.reshape(1, 12)))
print(np.transpose(array))  # 矩阵转置
print(array.T)  # T 一定要大写
print(array.T.dot(array))
print(array.clip(5, 9))  # 小于5的数变为5， 大于9的数变为9
'''

'''
# numpy 的索引应用
array = np.arange(2, 12).reshape((2, 5))
print(array)
print(array[1])  # 直接输出某一行
print(array[0][4])
print(array[0, 4])
print(array[:, 3])

# for循环依次输出array每一行
for row in array:
    print(row)

# for循环依次输出array每一列
for column in array.T:
    print(column)

print(array.flat)
print(array.flatten())
for item in array.flatten():
    print(item)
for item in array.flat:
    print(item)
'''

'''
# numpy 的 array 合并
array1 = np.array([[1, 2, 3], [7, 8, 9]])
array2 = np.array([[4, 5, 6], [10, 11, 12]])
array3 = np.vstack((array1, array2))  # vertical stack 垂直合并
print(array1, array2, '\n', array3)
print(array1.shape, array3.shape)
array4 = np.hstack((array1, array2))  # horizontal stack 水平合并
print(array1, array2, '\n', array4)
print(array1.shape, array4.shape)

# 对于一个一维的数列不能通过.T将其变为一列的矩阵
print(array1.T)
# 可以重新reshape，但很麻烦
print(array1.reshape((array1.size, 1)))
# 可以给array加一个维度
print(array1[:, np.newaxis])
print(array1[:, np.newaxis].shape)
# 定义的时候可以直接定义列向量
array5 = np.array([7, 8, 9])[:, np.newaxis]
print(array5)


# axis = 0,垂直合并
print(np.concatenate((array1, array2, array2, array1), axis=0))
# axis = 1,水平合并
print(np.concatenate((array1, array2, array2, array1), axis=1))
'''

'''
# array 的分割
array = np.arange(12).reshape((3, 4))
print(array)
# axis = 1, 是对行进行切分（垂直切），切成两半
print(np.split(array, 2, axis=1))
# axis = 0, 是对列进行切分（水平切），切成三份
print(np.split(array, 3, axis=0))
# 不等分割
print(np.array_split(array, 3, axis=1))

print(array.vsplit(2))
'''

# array 的赋值
a1 = np.arange(4)
a2 = a1
a3 = a2
a4 = a1.copy()
print(a1)
# 改变a1, a2, a3中的任意一个值，其他两个数列都会改变，但a4不会发生改变
a2[0] = 4
print(a1)
print(a1 is a3)
print(a4)


