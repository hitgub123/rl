import tensorflow as tf
from tensorflow import keras
from keras.layers import *
import numpy as np

tf.random.set_seed(1)
np.random.seed(1)

x_input = Input(shape=(None, 1))
y_input = Input(shape=(None, 1))

ki=keras.initializers.Constant(value=2)
bi=keras.initializers.Constant(value=1)
y = Dense(1, kernel_initializer=ki,bias_initializer=bi)(x_input)

def myloss(yt,yp):
    return tf.reduce_mean(tf.pow(yt - yp, 2))

# opt = tf.compat.v1.train.AdamOptimizer(1e-2)
opt = tf.compat.v1.train.GradientDescentOptimizer(1e-2)

m = keras.models.Model(inputs=[x_input, y_input], outputs=y)
m.compile(loss=myloss,optimizer=opt)


x_pure = np.random.randint(-10, 100, 320).astype(np.float).reshape(-1, 1)
x_train = x_pure
y_train = 3 * x_pure + 2

# 训练方法1
# for x_,y_ in zip(x_train,y_train):
#     m.fit([x_,y_],y_,verbose=1)

# 训练方法2
m.fit(x=[x_train, y_train], y=y_train, verbose=1, epochs=1,shuffle=False)

# 训练方法3
# m.train_on_batch(x=[x_train, y_train], y=y_train)

testdata = np.arange(3).reshape(-1, 1)
print(m.predict([testdata, testdata]))
