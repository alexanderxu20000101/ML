
import torch
import numpy as np
from torch.autograd import Variable

##################
data = [-1,-2,1,2]
tensor = torch.FloatTensor(data)   #
print(
    '\nabs',
    '\nnumpy:',np.abs(data),             #abs:绝对值， sin  ，cos，
    '\ntorch:',torch.abs(tensor)    #torch.mm    mm= matmul（矩阵乘法）
)


####################
data1 = [[1,2],[3,4]]
tensor1 = torch.FloatTensor(data1)   #
print(
    '\nabs',
    '\nnumpy:',np.matmul(data1,data1),             #abs:绝对值， sin  ，cos，
    '\ntorch:',torch.mm(tensor1,tensor1)    #torch.mm    mm= matmul（矩阵乘法）
)

###################

tensor2 =torch.FloatTensor([[1,2],[3,4]])