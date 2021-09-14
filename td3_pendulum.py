import tensorflow as tf
from tensorflow import keras
import numpy as np
import gym

#####################  hyper parameters  ####################

MAX_EPISODES = 120
MAX_EP_STEPS = 200

MEMORY_CAPACITY = 10000
GAME = 'Pendulum-v0'

np.random.seed(1)
tf.random.set_seed(1)


###############################  TD3  ####################################

class TD3(object):
    def _build_a(self):
        input1 = keras.layers.Input(shape=(self.s_dim,))
        net = keras.layers.Dense(30, activation='relu')(input1)
        net = keras.layers.Dense(self.a_dim, activation='tanh')(net)
        net = tf.multiply(net, self.a_bound)
        model = keras.models.Model(inputs=input1, outputs=net)
        optimizer = keras.optimizers.Adam(learning_rate=self.LR_A)
        model.compile(loss=self.my_loss, optimizer=optimizer)
        return model

    def _build_c(self):
        input1 = keras.layers.Input(shape=(self.s_dim,))
        input2 = keras.layers.Input(shape=(self.a_dim,))

        net1 = keras.layers.Dense(30)(input1)
        net2 = keras.layers.Dense(30)(input2)

        # 两个输入计算后合并
        net = keras.layers.Activation('relu')(net1 + net2)
        net = keras.layers.Dense(1)(net)
        model = keras.models.Model(inputs=[input1, input2], outputs=net)
        optimizer = keras.optimizers.Adam(learning_rate=self.LR_C)
        model.compile(loss='mse', optimizer=optimizer)
        return model

    def __init__(self, a_dim, s_dim, a_bound):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.TAU = 0.01  # soft replacement
        self.LR_A = 0.001  # learning rate for actor
        self.LR_C = 0.002  # learning rate for critic
        self.GAMMA = 0.9  # reward discount

        self.BATCH_SIZE = 32

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,

        self.model_a_e = self._build_a()
        self.model_a_t = self._build_a()
        self.model_a_t.set_weights(self.model_a_e.get_weights())

        self.model_c_e1 = self._build_c()
        self.model_c_t1 = self._build_c()
        self.model_c_t1.set_weights(self.model_c_e1.get_weights())

        self.model_c_e2 = self._build_c()
        self.model_c_t2 = self._build_c()
        self.model_c_t2.set_weights(self.model_c_e2.get_weights())

        self.update_cnt = 0
        self.update_a_cnt = 3  # 每更新critic update_a_cnt次，更新一次actor

        self.eval_noise_scale = .5  # a_target用s_计算a_后加上噪音

    # 每次学习前替换参数，把target的参数往eval偏TAU(1%)
    def soft_replace(self):
        ae_weights = self.model_a_e.get_weights()
        at_weights = self.model_a_t.get_weights()

        ce1_weights = self.model_c_e1.get_weights()
        ct1_weights = self.model_c_t1.get_weights()

        ce2_weights = self.model_c_e2.get_weights()
        ct2_weights = self.model_c_t2.get_weights()

        for i in range(len(at_weights)):
            at_weights[i] = self.TAU * ae_weights[i] + (1 - self.TAU) * at_weights[i]

        for i in range(len(ct1_weights)):
            ct1_weights[i] = self.TAU * ce1_weights[i] + (1 - self.TAU) * ct1_weights[i]

        for i in range(len(ct2_weights)):
            ct2_weights[i] = self.TAU * ce2_weights[i] + (1 - self.TAU) * ct2_weights[i]

        self.model_a_t.set_weights(at_weights)
        self.model_c_t1.set_weights(ct1_weights)
        self.model_c_t2.set_weights(ct2_weights)

    def choose_action(self, s, var=0):
        a = self.model_a_e(s[np.newaxis, :])[0]
        # 算出动作后，加上探索噪音进行随机探索。随训练进行逐渐减少杂音
        if var:
            a = np.clip(np.random.normal(a, var), -2, 2)  # add randomness to action selection for exploration
        return a

    def add_noise(self, a, clip=True):
        noise = tf.random.normal(shape=a.shape, mean=0.0, stddev=1.0, )
        noise = noise * self.eval_noise_scale
        if clip:
            noise = tf.clip_by_value(noise, -2 * self.eval_noise_scale, 2 * self.eval_noise_scale)
        a = a + noise
        a = tf.clip_by_value(a, -self.a_bound, self.a_bound)
        return a

    # actor的loss是critic的输出，这里把状态当y_true，行为当y_pre传给critic来计算
    # critic的输出越大，说明actor选择的行为越好，所以可以把critic的输出的负值当loss来最小化
    def my_loss(self, y_true, y_pre):
        loss1 = self.model_c_e1([y_true, y_pre])
        loss2 = self.model_c_e2([y_true, y_pre])
        loss = tf.minimum(loss1, loss2)
        loss = -tf.reduce_mean(loss)
        return loss



    def learn(self):
        self.update_cnt += 1

        indices = np.random.choice(MEMORY_CAPACITY, size=self.BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]
        ba_ = self.model_a_t(bs_)
        ba_ = self.add_noise(ba_)

        bq_1 = self.model_c_t1([bs_, ba_])
        bq_2 = self.model_c_t2([bs_, ba_])
        bq_ = tf.minimum(bq_1, bq_2)
        bq_target = br + self.GAMMA * bq_

        self.model_c_e1.fit([bs, ba], bq_target, verbose=0)
        self.model_c_e2.fit([bs, ba], bq_target, verbose=0)

        if self.update_cnt % self.update_a_cnt == 0:
            # 这里把状态bs当y_true传给actor的loss函数
            self.model_a_e.fit(bs, bs, verbose=0)
            self.soft_replace()

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1


###############################  training  ####################################

env = gym.make(GAME).unwrapped
env.seed(1)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high
var = 3  # control exploration
model = TD3(a_dim, s_dim, a_bound)

for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        a = model.choose_action(s, var)
        s_, r, done, info = env.step(a)
        model.store_transition(s, a, r / 10, s_)

        if model.pointer > MEMORY_CAPACITY:
            var *= .9995  # decay the action randomness
            model.learn()

        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS - 1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            break

while 1:
    s = env.reset()
    for t in range(200):
        env.render()
        s, r, done, info = env.step(model.choose_action(s))
        if done:
            break
