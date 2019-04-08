import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("MNIST_data",one_hot=True)

batch_size =100   #每次从mnist中取100张图,喂给神经网络训练
n_batch =mnist.train.num_examples   #所有mnist的数据有多少个(6w)

x=tf.placeholder(tf.float32,[None,784])     #none表示不确定. 每一张图是784.
                                            # 每次喂多少张图,不确定,因此用none
y=tf.placeholder(tf.float32,[None,10])      #道理路上

W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
prediction =tf.nn.softmax(tf.matmul(x,W)+b)

#loss= tf.reduce_mean(tf.square(y-prediction))
loss=tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction)
                    #这里loss用交叉上,

train_step =tf.train.GradientDescentOptimizer(0.2).minimize(loss)
                    #梯度下降计算 loss

init=tf.global_variables_initializer()
correct_prediction =tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
accuracy =tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})

        acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("iter"+str(epoch)+ ",Testing Accuracy "+ str(acc))
