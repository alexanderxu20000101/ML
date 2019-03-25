#通用性的模型
from keras.layers import  Input
from keras.layers import Dense
from keras.models import  Model
from keras.datasets import mnist
from keras.utils import to_categorical

#=========【1】数据准备=======
#从 系统自带的mnist中获得数据
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()  #把mnist的数据下载，放在 4个变量中

#从获得mnist的数据，进行压缩处理，得到的数据我们用于训练神经网络
train_images=train_images.reshape((60000,28*28))     #原图是28*28 彩色的，现在变化为 灰度图片。
train_images = train_images.astype('float32')/255   #归一化，就是确保数字在0和1之间。
test_images=test_images.reshape((10000,28*28))
test_images=test_images.astype('float32')/255
train_labels=to_categorical(train_labels)       #处理标签
test_lables = to_categorical(test_labels)

#=======【2】构建神经网络（前向传播）========
input = Input(shape=(784,))
x=Dense(64,activation='relu')(input)
x=Dense(64,activation='relu')(x)
x=Dense(64,activation='relu')(x)            #非常方便，想加多少层，就加多少层。
#x=Dense(64,activation='relu')(x)
#x=Dense(64,activation='relu')(x)
y=Dense(10,activation='softmax')(x)
model=Model(inputs=input,outputs=y)
#设置一些参数：优化器， 损失函数处理方式，
model.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()   #显示网络的结构

#=======【3】训练网络（用训练数据）（反向传播）=======
#喂数据为网络训练（把处理好的数据  image图片和label标签，喂给网络进行训练）  （反向传播）
#model.fit(train_images,train_labels)
model.fit(train_images,train_labels,epochs=1,batch_size=128)   # epochs  测试数据5w数据做一次为一轮epochs.
                            # batch_size 为每次从测试数据5w数据拿128个数据给系统训练


#=====【4】用评估数据评估效果（用测试数据）========
#评估测试数据集的效果（用测试数据集，看看效果如何）
test_loss,test_acc=model.evaluate(test_images,test_labels)
print('test_loss=>',test_loss)
print('test_acc=》',test_acc)

