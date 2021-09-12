import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.layers import *
tf.random.set_seed(1)
np.random.seed(1)

x=Input((1,))
out=Dense(1)(x)
out=Dense(1)(out)
m=keras.models.Model(x,out)

opt = tf.compat.v1.train.GradientDescentOptimizer(1)
ws=m.trainable_weights

x=np.array([[1]])
y=x*2+1

def getloss(y_true,y_pre):
    # 类型是Tensor，用np计算会报错 'numpy.ndarray' object has no attribute '_id'
    # return np.mean(np.square(y_true-y_pre),axis=-1)
    return tf.reduce_mean(tf.square(y_true - y_pre))

def train(x,y):
    with tf.GradientTape() as tape:
        y_pred = m(x)
        loss = getloss(y, y_pred)
        print('training\t',x,y,loss.numpy())
    grads = tape.gradient(loss, ws)
    grads_and_vars = zip(grads, ws)
    opt.apply_gradients(grads_and_vars)

for i in ws:
    print(i.numpy())
# m.summary()
# m.fit(x,y)
train(x,y)
print('>'*88)
for i in ws:
    print(i.numpy())
'''
输出如下：
[[-1.1600207]]
[0.]
[[0.03501177]]
[0.]
training	 [[1]] [[3]] 9.245336
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
[[-0.9471061]]
[0.21291457]
[[-7.0193396]]
[6.0812287]



运算如下：
x=1 y=3
layer1:-1.1600207
layer2:-1.1600207*0.03501177=-3.040614377943639

loss 9.245336
a=0.03501177   b=0    x=-1.1600207     y=3
y=ax+b 					
(ax+b-y)**2=       aaxx+(b-y)(b-y)+2(b-y)ax       =     aaxx+bb+yy-2by+2bax-2yax
dl/da=2xxa+2(b-y)x=  2*0.03501177*-1.1600207*-1.1600207 + 2*-1.1600207 *-3  =   7.0543512382644895
dl/db=2b+2ax-2y=   0.03501177*2*-1.1600207-2*3   =    -6.081228755887278
dl/dx=2aax+2ba-2ya= 2*0.03501177*0.03501177*-1.1600207-2*0.03501177*3 = -0.2129145825185115

a=(0.03501177-7.0543512382644895)*1= -7.01933946826449
b=(0--6.081228755887278)*1=6.081228755887278


loss -0.2129145825185115
c=-1.1600207   d=0    x=1    y=-1.1600207 
y=cx+d

dl/dc=dl/dy*dy/dc= -0.2129145825185115*x = -0.2129145825185115
dl/dd=dl/dy*dy/dd= -0.2129145825185115*1 = -0.2129145825185115
c=-1.1600207--0.2129145825185115 = -0.9471061174814885
d=0--0.2129145825185115 = 0.2129145825185115
'''