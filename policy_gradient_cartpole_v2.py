import gym,numpy as np
import tensorflow as tf,time
from tensorflow.keras import *

np.random.seed(1)
tf.random.set_seed(1)

#
def my_loss(y_true, y_pred):
    neg_log_prob=losses.categorical_crossentropy(y_true,y_pred)

    # y_true已经经过softmax处理，不需要用softmax_cross_entropy_with_logits
    # neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)

    # 下一行reduce_mean的代码没用，如果fit时传了权重，反而会出问题
    # neg_log_prob = tf.reduce_mean(neg_log_prob)

    return neg_log_prob

class PGModel(tf.keras.Model):
    def __init__(self,n_actions,n_features):
        super().__init__()
        self.dense1 = layers.Dense(units=10, input_dim=n_features, activation=activations.tanh)
        self.dense2 = layers.Dense(units=n_actions, activation=activations.softmax)

    '''
    Args:inputs: A tensor or list of tensors.
    Returns:A tensor if there is a single output, or a list of tensors if there are more than one outputs.
    '''
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

class PolicyGradient():
    def __init__(self,n_actions,n_features,learning_rate=1e-2,reward_decay=.99):
        self.n_actions = n_actions
        self.n_features = n_features

        self.ModelFP = 'models/model_diy.h5'
        self.save_model_episode = 1e2

        self.model = PGModel(n_actions,n_features)

        self.load_model()

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        self.gamma=reward_decay
        self.lr=learning_rate

        # 使用optimizers.Adam(self.lr) 600+次 到200
        self.optimizer=optimizers.Adam(self.lr)

        # 使用tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr) 1400+次到200
        # self.optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr)


        # self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer)
        # self.model.compile(loss=my_loss, optimizer=self.optimizer)
        self.model.add_loss(my_loss())              #3，add_loss
        self.model.compile(optimizer=self.optimizer)




    def choose_action(self, s):
        # prob = self.model.predict(s[np.newaxis, :])[0]
        # 不加.numpy()会报错probabilities do not sum to 1
        prob = self.model(s[np.newaxis, :])[0].numpy()
        action=np.random.choice(len(prob), p=prob)
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

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

    def reset_memory(self):
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

    def learn(self):
        s_batch = np.array(self.ep_obs)
        a_batch = np.array(self.ep_as,dtype=np.int32)
        r_batch = self._discount_and_norm_rewards()

        # learn里，model(s_batch) 比model.predict(s_batch)快1/3
        # choose_action里，model(s_batch) 比model.predict(s_batch)快好几倍

        # prob_batch = self.model.predict(s_batch) * tf.one_hot(a_batch, self.n_actions)
        '''
        正常应该用tf.one_hot(a_batch, self.n_actions)，最高rewrad=200+
        用self.model(s_batch) * tf.one_hot(a_batch, self.n_actions)能到700+
        估计是参数没设置好
        '''

        # 不清楚为什么要下一行代码，加上后效果更好
        # 个人感觉不该加上，因为r_s>0的行为需要奖励，r_s<0的行为需要奖惩罚，加上后逻辑就不对了
        # prob_batch = self.model(s_batch) * tf.one_hot(a_batch, self.n_actions)
        prob_batch = tf.one_hot(a_batch, self.n_actions)

        self.model.fit(s_batch, prob_batch, sample_weight=r_batch, verbose=0)


    def save_model(self):
        self.model.save_weights(self.ModelFP)
        print('>' * 88, 'model saved')

    def load_model(self):
        try:
            # 直接load_weights可能报错，传入dummy数据
            self.model(np.zeros((self.n_features,))[np.newaxis, :])
            self.model.load_weights(self.ModelFP)
            print('模型加载成功')
        except Exception as e:
            print('模型加载失败，',e)

def train(RL):
    DISPLAY_REWARD_THRESHOLD = 1e3  # renders environment if total episode reward is greater then this threshold
    RENDER = False  # rendering wastes time

    print(env.action_space)
    print(env.observation_space)
    print(env.observation_space.high)
    print(env.observation_space.low)

    for i_episode in range(3000):
        observation = env.reset()

        while True:
            if RENDER: env.render()
            action = RL.choose_action(observation)
            observation_, reward, done, _ = env.step(action)
            RL.store_transition(observation, action, reward)

            if done:
                ep_rs_sum = sum(RL.ep_rs)

                if 'running_reward' not in locals().keys():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
                if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering
                print("episode:", i_episode, "  reward:", int(running_reward))

                vt = RL.learn()
                RL.reset_memory()

                if running_reward>DISPLAY_REWARD_THRESHOLD or i_episode % RL.save_model_episode ==RL.save_model_episode- 1:
                    pass
                    # RL.save_model()
                break

            observation = observation_


def play(RL):
    for i_episode in range(3000):
        observation = env.reset()

        while 1:
            env.render()

            action = RL.choose_action(observation)
            observation_, reward, done, _ = env.step(action)
            RL.store_transition(observation, action, reward)

            if done:
                ep_rs_sum = sum(RL.ep_rs)
                if 'running_reward' not in locals().keys():
                # if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
                print("episode:", i_episode, "  reward:", int(running_reward))
                RL.reset_memory()
                break

            observation = observation_

env = gym.make('CartPole-v0')
env.seed(1)  # reproducible, general Policy gradient has high variance
env = env.unwrapped
RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=1e-2,
    reward_decay=0.99,
)

train(RL)
# play(RL)