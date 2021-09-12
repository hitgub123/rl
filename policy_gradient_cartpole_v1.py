import gym, numpy as np
import tensorflow as tf
from tensorflow import keras

np.random.seed(1)
tf.random.set_seed(1)

'''
policy_gradient_cartpole_v1:能对多个模型输入使用3种loss，

policy_gradient_cartpole_v2:能对多个模型输入使用2种loss
categorical_crossentropy会对y_pred减去reduce_sum(y_pred)，不会出现inf/nan，
训练可能出现和tf.reduce_sum(-tf.math.log(y_pred) * y_true, axis=1)不一样的情况，
自定义函数时如果部分代码可以换成自带的函数，推荐全部用自带的
'''
def my_loss(y_true, y_pred):
    # loss = tf.reduce_sum(-tf.math.log(y_pred) * y_true, axis=1)
    loss = keras.losses.categorical_crossentropy(y_true, y_pred)

    # 下一行reduce_mean的代码没用，如果fit时传了权重，反而会出问题
    # loss = tf.reduce_mean(loss)
    return loss


def my_loss_weights(y_true, y_pred, tf_vt):
    loss = keras.losses.categorical_crossentropy(y_true, y_pred)
    loss = loss * tf_vt
    return loss


class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate,
            reward_decay,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self.model = self._build_net()

    def _build_net(self):
        tf_obs = tf.keras.layers.Input(shape=(self.n_features,))
        tf_acts = tf.keras.layers.Input(shape=(2,))
        tf_vt = tf.keras.layers.Input(shape=(1,))

        # fc1
        layer = keras.layers.Dense(
            units=10,
            activation=keras.activations.tanh,
            name='layer',
        )(tf_obs)

        # fc2
        all_act_prob = keras.layers.Dense(
            units=self.n_actions,
            activation=keras.activations.softmax,
            name='all_act_prob',
        )(layer)

        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr)
        model = keras.models.Model(inputs=(tf_obs, tf_acts, tf_vt), outputs=all_act_prob)

        '''
        定义loss的方法：1，默认的；2，loss=自定义loss；3，add_loss(可以添加其他参数)
        add_loss添加和其他两种一样的损失函数时，运行结果不一样，原因不明(源码看不懂)。
        这个例子add_loss学习速度更慢，但效果更好。前两个reward=700+，第3个1000+
        fit的sample_weight对方法1/2有效，对3似乎无效
        '''
        # model.compile(loss='categorical_crossentropy', optimizer=self.optimizer)          #1，默认的
        # model.compile(loss=my_loss, optimizer=self.optimizer)                             #2，loss=自定义loss
        model.add_loss(my_loss_weights(tf_acts, all_act_prob, tf_vt))                       #3，add_loss(可以添加其他参数)
        model.compile(loss=None,optimizer=self.optimizer)

        return model

    def choose_action(self, observation):
        # prob_weights = self.model.predict((observation[np.newaxis, :],np.ones((1,2)), np.ones((1,1))))
        prob_weights = self.model((observation[np.newaxis, :], np.ones((1, 2)), np.ones((1, 1))))[0].numpy()
        action = np.random.choice(len(prob_weights), p=prob_weights)
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # discount and normalize episode reward
        r_s = self._discount_and_norm_rewards()
        ob_s = np.vstack(self.ep_obs)
        a_s = tf.one_hot(np.array(self.ep_as), self.n_actions)

        # 不清楚为什么要下一行代码，加上后效果更好
        # 个人感觉不该加上，因为r_s>0的行为需要奖励，r_s<0的行为需要奖惩罚，加上后逻辑就不对了
        # a_s = self.model((ob_s, np.ones((len(self.ep_obs), 2)), np.ones((len(self.ep_obs), 1)))) * a_s

        self.model.fit([ob_s, a_s, r_s], a_s, verbose=0, sample_weight=r_s)
        # self.model.fit([ob_s, a_s, r_s],a_s, verbose=0)

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []  # empty episode data

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs


DISPLAY_REWARD_THRESHOLD = 2e3  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time

env = gym.make('CartPole-v0')
env.seed(1)  # reproducible, general Policy gradient has high variance
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=1e-2,
    reward_decay=0.99,
)

for i_episode in range(3000):
    observation = env.reset()

    while True:
        if RENDER: env.render()
        action = RL.choose_action(observation)
        observation_, reward, done, _ = env.step(action)
        RL.store_transition(observation, action, reward)

        if done:
            ep_rs_sum = sum(RL.ep_rs)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))

            vt = RL.learn()
            break

        observation = observation_
