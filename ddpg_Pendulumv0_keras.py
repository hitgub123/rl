import tensorflow as tf
from tensorflow import keras
import numpy as np
import gym

#####################  hyper parameters  ####################

MAX_EPISODES = 200
MAX_EP_STEPS = 200

MEMORY_CAPACITY = 10000

RENDER = False
ENV_NAME = 'Pendulum-v0'

np.random.seed(1)
tf.random.set_seed(1)

###############################  DDPG  ####################################

class DDPG(object):
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
        # net = tf.nn.relu(net1 + net2)     #效果不好

        # net = keras.layers.concatenate([net1, net2])
        # net = keras.layers.Dense(30, activation='relu')(net)

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

        self.model_c_e = self._build_c()
        self.model_c_t = self._build_c()
        self.model_c_t.set_weights(self.model_c_e.get_weights())

    # 每次学习前替换参数，把target的参数往eval偏TAU(1%)
    def soft_replace(self):
        ae_weights = self.model_a_e.get_weights()
        at_weights = self.model_a_t.get_weights()

        ce_weights = self.model_c_e.get_weights()
        ct_weights = self.model_c_t.get_weights()

        for i in range(len(at_weights)):
            at_weights[i] = self.TAU * ae_weights[i] + (1 - self.TAU) * at_weights[i]

        for i in range(len(ct_weights)):
            ct_weights[i] = self.TAU * ce_weights[i] + (1 - self.TAU) * ct_weights[i]

        self.model_a_t.set_weights(at_weights)
        self.model_c_t.set_weights(ct_weights)

    def choose_action(self, s):
        a = self.model_a_e(s[np.newaxis, :])[0]
        return a

    # actor的loss是critic的输出，这里把状态当y_true，行为当y_pre传给critic来计算
    # critic的输出越大，说明actor选择的行为越好，所以可以把critic的输出的负值当loss来最小化
    def my_loss(self,y_true,y_pre):
        loss=self.model_c_e([y_true,y_pre])
        loss=-tf.reduce_mean(loss)
        return loss

    def learn(self):
        self.soft_replace()

        indices = np.random.choice(MEMORY_CAPACITY, size=self.BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        ba_ = self.model_a_t(bs_)
        bq_ = self.model_c_t([bs_, ba_])
        bq_target = br + self.GAMMA * bq_

        self.model_c_e.fit([bs, ba], bq_target,verbose=0)
        # 这里把状态bs当y_true传给actor的loss函数
        self.model_a_e.fit(bs,bs,verbose=0)

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1


###############################  training  ####################################

env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high

ddpg = DDPG(a_dim, s_dim, a_bound)

var = 3  # control exploration
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()

        # Add exploration noise
        a = ddpg.choose_action(s)
        # 算出动作后，加上噪音进行随机探索。随训练进行逐渐减少杂音
        a = np.clip(np.random.normal(a, var), -2, 2)  # add randomness to action selection for exploration
        s_, r, done, info = env.step(a)

        ddpg.store_transition(s, a, r / 10, s_)

        if ddpg.pointer > MEMORY_CAPACITY:
            var *= .9995  # decay the action randomness
            ddpg.learn()

        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS - 1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            if ep_reward > -300: RENDER = True
            break
