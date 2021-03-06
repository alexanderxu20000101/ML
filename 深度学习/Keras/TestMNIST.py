#测试 mnist，把里面的图片信息显示出来。
from keras.datasets import mnist

import  matplotlib.pyplot as plt     #画图的包

(train_images,train_labels),(test_images,test_labels)=mnist.load_data()  #把mnist的数据下载，放在 4个变量中


print(train_labels[1])
print(plt.imshow(train_images[1]))
plt.imshow(train_images[1])  #显示  训练数据中的第一张图
train_images[1]      #把测试数据第一张图的矩阵数据显示出来
print(train_images.shape)
print(len(train_labels))
print(train_labels[:15])   #显示 测试数据的标签的第0  到15个.



#################################################
from  keras  import models
from keras import layers
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.utils import plot_model

#从 系统自带的mnist中获得数据
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()  #把mnist的数据下载，放在 4个变量中
#从获得mnist的数据，进行压缩处理，得到的数据我们用于训练神经网络
train_images=train_images.reshape((60000,28*28))     #原图是28*28 彩色的，现在变化为 灰度图片。
train_images = train_images.astype('float32')/255   #归一化，就是确保数字在0和1之间。
test_images=test_images.reshape((10000,28*28))
test_images=test_images.astype('float32')/255
train_labels=to_categorical(train_labels)       #处理标签
test_lables = to_categorical(test_labels)

#创建一个神经网络的模型
network = models.Sequential()
network.add(layers.Dense(512,activation='relu',input_shape=(28*28,))) #全连接输入层，512个节点，输入参数28*28个。
network.add(layers.Dense(10,activation='softmax'))           #输出层，激活函数为softmax

network.compile(optimizer='rmsprop',     #优化器
                loss='categorical_crossentropy',    #损失函数
                metrics=['accuracy'])     # 精确 评估方式
network.summary()
plot_model(network,to_file='model_testMNIST.png')


#把处理好的数据  image图片和label标签，喂给网络进行训练
history=network.fit(train_images,train_labels,epochs=1,batch_size=128)

# Plot training & validation accuracy values
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


#######################################
#评估测试数据集的效果（用测试数据集，看看效果如何）
test_loss,test_acc=network.evaluate(test_images,test_labels)
print('test_loss=>',test_loss)
print('test_acc=》',test_acc)
