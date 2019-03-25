
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module): # 必须集成nn.Module
   def __init__(self):
        super(Net, self).__init__() # 必须调用父类的构造函数，传入类名和self
                # 输入是1个通道(灰度图)，卷积feature
                # map的个数是6，大小是5x5，无padding，stride是1。
        self.conv1 = nn.Conv2d(1, 6, 5)

        # 第二个卷积层feature map个数是16，大小还是5*5，无padding，stride是1。
        self.conv2 = nn.Conv2d(6, 16, 5)

        # 仿射层y = Wx + b，ReLu层没有参数，因此不在这里定义
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

def forward(self, x):
   # 卷积然后Relu然后2x2的max pooling
   x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
   # 再一层卷积relu和max pooling
   x = F.max_pool2d(F.relu(self.conv2(x)), 2)
   # 把batch x channel x width x height 展开成batch x all_nodes
   x = x.view(-1, self.num_flat_features(x))
   x = F.relu(self.fc1(x))
   x = F.relu(self.fc2(x))
   x = self.fc3(x)
   return x

def num_flat_features(self, x):
   size = x.size()[1:] # 除了batchSize之外的其它维度
   num_features = 1
   for s in size:
       num_features *= s
   return num_features

net = Net()

print(net)