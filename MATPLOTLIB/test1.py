#NumPy Matplotlib | 菜鸟教程 http://www.runoob.com/numpy/numpy-matplotlib.html

import numpy as  np
from matplotlib import pyplot as plt

x=np.arange(1,11)    # x从1~10  之间. x坐标
y= 2*x+5

# plt.subplot(2,1,1)   #画第一个子图.

fig=plt.figure(1)    #第一张图
plt.title("matplotlib demot图形的总标题")
plt.xlabel("x axis caption  x轴的标题")
plt.ylabel("y  axis caption  y轴的标题")
plt.plot(x,y,"ob")    # 画线     ob:用原点 显示.而不是线条.

fig2=plt.figure(2)   #第二张图
plt.plot(x,3*x)    #画一条线
plt.plot(x,x*x)    #再画一条线.

# plt.subplot(2,1,2)    #画第二个子图
# plt.title("matplotlib demot图形的总标题2")
# plt.xlabel("x axis caption  x轴的标题2")
# plt.ylabel("y  axis caption  y轴的标题2")
# plt.plot(x,y,color="red",linewidth=2.5)    # 画线     ob:用原点 显示.而不是线条.

plt.show()
