import multiprocessing
import threading
import tensorflow as tf
from tensorflow import keras
from keras.layers import *
import numpy as np
import gym

np.random.seed(1)
tf.random.set_seed(1)

GAME = 'CartPole-v0'
N_WORKERS = multiprocessing.cpu_count()
MAX_GLOBAL_EP = 5000
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
ENTROPY_BETA = 0.001
LR_A = 1e-3  # learning rate for actor
LR_C = 1e-2  # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0

env = gym.make(GAME)
env.seed(1)
N_S = env.observation_space.shape[0]
N_A = env.action_space.n

OPT_A = keras.optimizers.Adam(LR_A)
OPT_C = keras.optimizers.Adam(LR_C)


class ACNet(object):
    def __init__(self, OPT_A=OPT_A, OPT_C=OPT_C):
        self.OPT_A = OPT_A
        self.OPT_C = OPT_C

        self.a_model, self.c_model = self._build_net()

    def _build_net(self):
        w_init = tf.random_normal_initializer(0., .1)

        a_s = Input(shape=(N_S,))
        a_layer = Dense(20, activation='relu', kernel_initializer=w_init)(a_s)
        a_prob = Dense(N_A, activation='softmax', kernel_initializer=w_init)(a_layer)
        a_model = keras.models.Model(inputs=a_s, outputs=a_prob)

        c_s = Input(shape=(N_S,))
        c_layer = Dense(20, activation='relu', kernel_initializer=w_init)(c_s)
        v = Dense(1, kernel_initializer=w_init)(c_layer)
        c_model = keras.models.Model(inputs=c_s, outputs=v)

        return a_model, c_model

    def choose_action(self, s):  # run by a local
        prob_weights = self.a_model(s[np.newaxis, :])[0].numpy()
        a = np.random.choice(range(prob_weights.shape[-1]), p=prob_weights)
        return a


class Worker(object):
    def __init__(self, name, globalAC):
        self.env = gym.make(GAME).unwrapped
        self.name = name
        self.globalAC = globalAC
        self.AC = ACNet()
        self.render = False

    def pull_global(self):
        self.AC.a_model.set_weights(self.globalAC.a_model.get_weights())
        self.AC.c_model.set_weights(self.globalAC.c_model.get_weights())

    def update_global(self, buffer_s, buffer_a, buffer_v_target):
        with tf.GradientTape(persistent=True) as tape:
            a_prob = self.AC.a_model(buffer_s)
            buffer_v_pred = self.AC.c_model(buffer_s)

            # a - b 等于 tf.subtract(a, b)
            td = buffer_v_target - buffer_v_pred
            # td = tf.subtract(buffer_v_target, buffer_v_pred)

            # 以前ac代码的aloss，也能收敛
            # log_prob = tf.reduce_sum(tf.math.log(a_prob) * tf.one_hot(buffer_a, N_A, dtype=tf.float32), axis=1,
            #                          keepdims=True)
            # exp_v = log_prob * tf.stop_gradient(td)
            # exp_v = log_prob * td

            # 莫烦的a3c的代码的aloss
            log_prob = tf.reduce_sum(tf.math.log(a_prob + 1e-5) * tf.one_hot(buffer_a, N_A, dtype=tf.float32), axis=1,keepdims=True)
            exp_v = log_prob * tf.stop_gradient(td)
            entropy = -tf.reduce_sum(a_prob * tf.math.log(a_prob + 1e-5),axis=1, keepdims=True)  # encourage exploration
            exp_v = ENTROPY_BETA * entropy + exp_v

            a_loss = tf.reduce_mean(-exp_v)

            # 3个loss都能收敛
            # c_loss = tf.reduce_mean(keras.losses.mse(buffer_v_target, buffer_v_pred))
            # c_loss = keras.losses.mse(buffer_v_target, buffer_v_pred)
            c_loss = tf.reduce_mean(tf.square(td))

        # print(a_loss.numpy(),c_loss.numpy())
        a_grads = tape.gradient(a_loss, self.AC.a_model.trainable_weights)
        a_grads_and_vars = zip(a_grads, self.globalAC.a_model.trainable_weights)
        self.globalAC.OPT_A.apply_gradients(a_grads_and_vars)

        c_grads = tape.gradient(c_loss, self.AC.c_model.trainable_weights)
        c_grads_and_vars = zip(c_grads, self.globalAC.c_model.trainable_weights)
        self.globalAC.OPT_C.apply_gradients(c_grads_and_vars)

        del tape

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            while True:
                if self.name == 'W_0' and self.render:
                    self.env.render()
                a = self.AC.choose_action(s)
                s_, r, done, info = self.env.step(a)
                if done: r = -5

                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    if done:
                        v_s_ = 0  # terminal
                    else:
                        v_s_ = self.AC.c_model(s_[np.newaxis, :])[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:  # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(
                        buffer_v_target)

                    self.update_global(buffer_s, buffer_a, buffer_v_target)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.pull_global()

                s = s_
                total_step += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.99 * GLOBAL_RUNNING_R[-1] + 0.01 * ep_r)
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %i" % GLOBAL_RUNNING_R[-1],
                    )
                    GLOBAL_EP += 1
                    if GLOBAL_RUNNING_R[-1] > 200: self.render = True
                    if GLOBAL_EP % 100 == 99: print('max', max(GLOBAL_RUNNING_R))
                    break


if __name__ == "__main__":

    with tf.device("/cpu:0"):

        GLOBAL_AC = ACNet()
        workers = []

        for i in range(N_WORKERS):
            i_name = 'W_%i' % i
            workers.append(Worker(i_name, GLOBAL_AC))

    COORD = tf.train.Coordinator()

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)