import  keras
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential     #sequential就是线性的意思，线性回归，按照顺序构成的模型。
from keras.layers import Dense    #全连接层（每个节点都和下一层的每一个节点连接）



x_data=np.random.rand(100)   # 用numpy生成100个随机点。

noise=np.random.normal(0,0.01,x_data.shape)   #没有noise，所有点都分布一条直线上。有，就会随机分布在直线的两边。
y_data=x_data*0.1+0.2+noise

plt.scatter(x_data,y_data)  #显示随机点。(闪点分布)
plt.show()



##########################################################


model = Sequential()  # 构建一个顺序的模型

model.add(Dense(units=1, input_dim=1))  # 在模型中，添加一个全连接层（参数2个，
# 意思：输入一个x，就输出一个y。dim就是一层，）

model.compile(optimizer='sgd', loss='mse')  # sgd:随机梯度下降法，Stonchastic gradient descent。
# mse：均方误差 Mean SquaredError


for step in range(3001):  # 训练3001次
    cost = model.train_on_batch(x_data, y_data)  # 每次训练一个批次
    if step % 500 == 0:  # 每500个batch打印一次cost的数值
        print('cost:', cost)

W, b = model.layers[0].get_weights()  # 打印权值和  偏置值
print('W:', W, 'b:', b)

y_pred = model.predict(x_data)  # x_data输入网络，得到预测值 y_pred
plt.scatter(x_data, y_data)  # 显示随机点
plt.plot(x_data, y_pred, 'r-', lw=3)
plt.show()



################################################