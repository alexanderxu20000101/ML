#序列模型

from keras.models  import Sequential
from keras.layers import Dense
from keras.layers import Activation


# #第一种 创建神经网络的方法
# layers =[Dense(32,input_shape=(784,)),
#         Activation('relu'),
#         Dense(10),
#         Activation('softmax')]
# model = Sequential(layers)
# model.summary()      #显示汇总信息。



from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG   #画出 网络结构图的包
import os
os.environ["PATH"] += os.pathsep + 'I:\Program Files (x86)\Graphviz2.38\bin'



#第二种：创建神经网络的方法
model=Sequential()     #创建一个空的，类似一个session。     然后一层  一层的加上
model.add(Dense(32,input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dense(10))
#model.add(Activation('softmax'))
model.summary()

#SVG(model_to_dot(model).create(prog='dot',format='svg'))


