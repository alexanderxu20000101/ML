#使用 RNN 进行情感分析的初学者指南 | 雷锋网 https://www.leiphone.com/news/201806/5QY2k0kEkf9d6xUA.html

from keras.datasets import imdb
vocabulary_size = 50      # 所有字符一共5000(故意改成50)
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = vocabulary_size)
            #从自带imdb中获取一批数据集.(训练和测试.)

print('Loaded dataset with {} training samples, {} test samples'.format(len(X_train), len(X_test)))

print('---review---')
print(X_train[6])     #显示第5个数据(第五篇的英文评论内容.这里显示的数字(数据集已经把英文字母转成数字))
print('---label---')
print(y_train[6])    #第五个数据的标签   0 :差评   1:好评

word2id = imdb.get_word_index()
id2word = {i: word for word, i in word2id.items()}
print('---review with words---')
print([id2word.get(i, ' ') for i in X_train[6]])   # 显示第五个数据的 英文内容
print('---label---')
print(y_train[6])

print('Maximum review length: {}'.format(
len(max((X_train + X_test), key=len))))     #评论中,最长的多长.

print('Minimum review length: {}'.format(
len(min((X_test + X_test), key=len))))    #数据(评论)中最短的多少.


from keras.preprocessing import sequence
max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)  # 对数据进行整理(都变成500个字符. 太长的截断. 太短用0的补pad)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)


from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
embedding_size=32
model=Sequential()   #序列方式.觉得部分都是这一种.  另外一种fucntion(运行节点之间任意连接,类似自由图)极少用
model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
                #输入层. 参数分别是. 输入参数个数, ?, 输入参数类型
model.add(LSTM(100))    #添加一个ltsm的隐含层. 100个节点.
model.add(Dense(1, activation='sigmoid'))       #添加一个输出层, 一个节点.激活函数sigmoid
print(model.summary())     #显示神经网络结构(简单显示)

model.compile(loss='binary_crossentropy',
 optimizer='adam',
 metrics=['accuracy'])
batch_size = 4   #64
num_epochs = 1
X_valid, y_valid = X_train[:batch_size], y_train[:batch_size]    #测试数据.
X_train2, y_train2 = X_train[batch_size:], y_train[batch_size:]
model.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=num_epochs)
                                #训练.validation 指定一些数据作为训练集.作用类似测试数据.
scores = model.evaluate(X_test, y_test, verbose=0)  #测试数据评估
print('Test accuracy:', scores[1])