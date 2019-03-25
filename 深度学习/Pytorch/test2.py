import torch

# N is batch size,  每次拿多少个数据喂给网络（如 6w数据，每次N（100）数据，轮600轮）。 N=2  ;
# D_in is input dimension;  输入层的节点个数
# H is hidden dimension  隐含层 节点个数; D_out is output dimension.  输出层节点个数
N, D_in, H, D_out = 64, 1000, 100, 10    #，输入层节点数，隐含层节点数，输出层节点数

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)     #输入的数据的结构，这里数据结构是向量tensor（可以看成，多维数组。目前数据是随机生成的）
y = torch.randn(N, D_out)    #输出的数据的结构

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
model = torch.nn.Sequential(    # 建立和定义一个2层的神经网络model
    torch.nn.Linear(D_in, H),       #定义输入层
    torch.nn.ReLU(),              #指定输入层 用的激活函数为relu
    torch.nn.Linear(H, D_out),     #定义输出层
)

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(reduction='sum')      #指定损失函数为  sum。根据项目 ，你的项目可能选用别的损失函数。

learning_rate = 1e-4            #指定学习率大小
for t in range(500):      #做500轮
    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Tensor of input data to the Module and it produces
    # a Tensor of output data.
    y_pred = model(x)    # 输入x放入  模型中， 得到实际的输出y  （取名字：y_pred，  y为输出）

    # Compute and print loss. We pass Tensors containing the predicted and true
    # values of y, and the loss function returns a Tensor containing the
    # loss.
    loss = loss_fn(y_pred, y)    #计算loss  = f(y-y~)   ,f函数在这里是sum
    print(t, loss.item())

    # Zero the gradients （剃度，就是斜率的意思） before running the backward pass.
    model.zero_grad()     #设置剃度斜率为0   ？？

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Tensors with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    # 根据所有参数计算损失函数loss的梯度斜率，参数存在tensor数据结构中（如果之前设置require——grad=true的话），
    loss.backward()     #反向传播，计算梯度，修正参数，参数存在tensor多维数组中。

    # Update the weights using gradient descent. Each parameter is a Tensor, so
    # we can access its gradients like we did before.   ?????
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad