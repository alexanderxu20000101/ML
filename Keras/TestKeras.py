from  keras  import models
from keras import layers
from keras.datasets import mnist
from keras.utils import to_categorical

#从 系统自带的mnist（6w张图片）中获得数据
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()  #把mnist的数据下载，放在 4个变量中

#从获得mnist的数据，进行压缩处理，得到的数据我们用于训练神经网络
train_images=train_images.reshape((60000,28*28))     #原图是28*28 彩色的，现在变化为 灰度图片。
train_images = train_images.astype('float32')/255   #归一化，就是确保数字在0和1之间。
test_images=test_images.reshape((10000,28*28))
test_images=test_images.astype('float32')/255
train_labels=to_categorical(train_labels)       #处理标签
test_lables = to_categorical(test_labels)



#创建一个神经网络的模型  （一个输入层，一个输出层，没哟隐含层）
network = models.Sequential()    #序列模型（最常见），另外一种是api（很少用）
network.add(layers.Dense(512,activation='relu',input_shape=(28*28,))) #输入参数28*28， 512个节点，激活函数 relu，dense 表示全连接层
network.add(layers.Dense(10,activation='softmax'))           #输出层：10个节点，激活函数为softmax（就是所谓分类）
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])      #评估方式为  精准评估

#喂数据为网络训练（把处理好的数据  image图片和label标签，喂给网络进行训练）  （正向传播+反向传播）
network.fit(train_images,train_labels,epochs=5,batch_size=128)  #每次拿128个数据，循环 5轮。

#评估测试数据集的效果（用测试数据集，看看效果如何）
test_loss,test_acc=network.evaluate(test_images,test_labels)
print('test_loss=>',test_loss)
print('test_acc=》',test_acc)
