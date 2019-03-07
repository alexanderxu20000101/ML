import  keras
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential     #sequential就是线性的意思，线性回归，按照顺序构成的模型。
from keras.layers import Dense    #全连接层（每个节点都和下一层的每一个节点连接）
from keras.optimizers import SGD



####################
x_data = np.linspace(-0.5,0.5,200)           #从-0,5  到 0.5   ,一共生成200个点
noise= np.random.normal(0,0.02,x_data.shape)
y_data= np.square(x_data)+noise

plt.scatter(x_data,y_data)   #显示随机点
plt.show()     #画成一个图形。



###################


model = Sequential()  # 构建一个顺序的模型

# 生成一个 1-10-1 的网络，中间是10节点的隐含层
model.add(Dense(units=10, input_dim=1))  # 在模型中，添加一个全连接层（参数2个，
# 意思：输入一个x，就输出一个y。dim就是一层，）
model.add(Dense(units=1))  # 增加中间那个  10个节点的隐含层

model.compile(optimizer='sgd', loss='mse')  # sgd:随机梯度下降法，Stonchastic gradient descent。
# mse：均方误差 Mean SquaredError
sgd = SGD()

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



###############