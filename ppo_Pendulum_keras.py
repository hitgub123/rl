import tensorflow as tf
from tensorflow import keras
from keras.layers import *
import numpy as np
import gym

np.random.seed(1)
tf.random.set_seed(1)

EP_MAX = 500
BATCH = 32
EP_LEN = 200
GAMMA = 0.9
A_LR = 0.0001
C_LR = 0.0002

A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
S_DIM, A_DIM = 3, 1
epsilon=0.2

env = gym.make('Pendulum-v0').unwrapped
env.seed(1)
a_bound = env.action_space.high[0]


class PPO(object):
    def __init__(self):
        self.opt_a = tf.compat.v1.train.AdamOptimizer(A_LR)
        self.opt_c = tf.compat.v1.train.AdamOptimizer(C_LR)

        self.model_a = self._build_anet(trainable=True)
        self.model_a_old = self._build_anet(trainable=False)
        self.model_c = self._build_cnet()

    def _build_anet(self,trainable=True):
        tfs_a = Input([S_DIM], )
        l1 = Dense(100, 'relu',trainable=trainable)(tfs_a)
        mu = a_bound * Dense(A_DIM, 'tanh',trainable=trainable)(l1)
        sigma = Dense(A_DIM, 'softplus',trainable=trainable)(l1)
        model_a = keras.models.Model(inputs=tfs_a, outputs=[mu, sigma])
        return model_a

    def _build_cnet(self):
        tfs_c = Input([S_DIM], )
        l1 = Dense(100, 'relu')(tfs_c)
        v = Dense(1)(l1)
        model_c = keras.models.Model(inputs=tfs_c, outputs=v)
        model_c.compile(optimizer=self.opt_c, loss='mse')
        return model_c

    def update(self, s, a, r):
        self.model_a_old.set_weights(self.model_a.get_weights())

        mu, sigma = self.model_a_old(s)
        oldpi = tf.compat.v1.distributions.Normal(loc=mu, scale=sigma)
        old_prob_a = oldpi.prob(a)

        v = self.get_v(s)
        adv = r - v

        for i in range(A_UPDATE_STEPS):
            with tf.GradientTape() as tape:
                mu, sigma = self.model_a(s)
                pi = tf.compat.v1.distributions.Normal(loc=mu, scale=sigma)
                ratio = pi.prob(a) / (old_prob_a + 1e-5)
                surr = ratio * adv
                x2 = tf.clip_by_value(ratio, 1. - epsilon, 1. + epsilon) * adv
                x3 = tf.minimum(surr, x2)
                aloss = -tf.reduce_mean(x3)

            a_grads = tape.gradient(aloss, self.model_a.trainable_weights)
            a_grads_and_vars = zip(a_grads, self.model_a.trainable_weights)
            self.opt_a.apply_gradients(a_grads_and_vars)

        self.model_c.fit(s, r, verbose=0, shuffle=False,epochs=C_UPDATE_STEPS)

    def choose_action(self, s):
        mu, sigma = self.model_a(s)
        pi = tf.compat.v1.distributions.Normal(loc=mu, scale=sigma)
        a = tf.squeeze(pi.sample(1), axis=0)
        return np.clip(a, -2, 2)

    def get_v(self, s):
        v = self.model_c(s)
        return v



ppo = PPO()
all_ep_r = []
for ep in range(EP_MAX):                    #train
    s = env.reset()
    buffer_s, buffer_a, buffer_r = [], [], []
    ep_r = 0
    s = np.reshape(s, (-1, S_DIM))
    for t in range(EP_LEN):  # in one episode
        a = ppo.choose_action(s)
        s_, r, done, _ = env.step(a)
        s_ = np.reshape(s_, (-1, S_DIM))
        buffer_s.append(s)
        buffer_a.append(a)
        buffer_r.append((r + 8) / 8)  # normalize reward, find to be useful
        s = s_
        ep_r += r

        # update ppo
        if (t + 1) % BATCH == 0 or t == EP_LEN - 1:
            v_s_ = ppo.get_v(s_)[0,0]
            discounted_r = []
            for r in buffer_r[::-1]:
                v_s_ = r + GAMMA * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()

            bs = np.vstack(buffer_s)
            ba = np.vstack(buffer_a)
            br = np.array(discounted_r)
            buffer_s, buffer_a, buffer_r = [], [], []
            ppo.update(bs, ba, br)
    if ep == 0:
        all_ep_r.append(ep_r)
    else:
        all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r * 0.1)
    print(
        'Ep: %i' % ep,
        "|Ep_r: %i" % ep_r,
    )

while 1:                        #play
    s = env.reset()
    for t in range(EP_LEN):
        s = s.reshape([-1, S_DIM])
        env.render()
        s, r, done, info = env.step(ppo.choose_action(s))
        if done:
            break
