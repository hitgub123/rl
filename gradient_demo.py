import tensorflow.compat.v1 as tf
import tensorflow as tf2

tf.disable_v2_behavior()
import numpy as np

tf2.random.set_seed(1)
sess = tf.Session()
np.random.seed(1)

x_input = tf.placeholder(tf.float32, name='x_input')
y_input = tf.placeholder(tf.float32, name='y_input')
w = tf.Variable(2.0, name='weight')
b = tf.Variable(-1.0, name='biases')
y = x_input * w + b

# loss_op = tf.reduce_mean(tf.pow(y_input - y, 2))
loss_op = tf.reduce_mean(y_input - y)

opt = tf.compat.v1.train.GradientDescentOptimizer(1)

# minimize的写法,   loss_op=y-wx-b >>>>  dw=-x   db=-1
gradients_node = tf.gradients(loss_op, [w, b])
train_op = opt.minimize(loss_op)

train_weights = [w, b]
# 写法1
# gradients_node = tf.gradients(loss_op, train_weights)
# gradients_node1 = zip(gradients_node,train_weights)
# train_op = opt.apply_gradients(gradients_node1)

# 写法2
# gradients_node = opt.compute_gradients(loss_op, train_weights)
# train_op = opt.apply_gradients(gradients_node)

init = tf.global_variables_initializer()
sess.run(init)

x_pure = np.random.randint(1, 4, 1)
x_train = x_pure
y_train = 3 * x_pure + 2
print(x_train, y_train)
print(sess.run([w, b]), '>>>')
for i in range(x_pure.size):
    _, gradients, loss = sess.run([train_op, gradients_node, loss_op],
                                  feed_dict={x_input: x_train[i], y_input: y_train[i]})
    print("epoch: {} \t loss: {} \t gradients: {}".format(i, loss, gradients))
print(sess.run([w, b]), '>>>')
print(sess.run(y, feed_dict={x_input: np.array([0, 1, 2])}))
sess.close()

'''
[2] [8]
[2.0, -1.0] >>>
epoch: 0 	 loss: 5.0 	 gradients: [(-2.0, 4.0), (-1.0, 0.0)]
[4.0, 0.0] >>>
[0. 4. 8.]
'''
