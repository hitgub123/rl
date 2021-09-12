import gym, numpy as np
import tensorflow as tf
from tensorflow import keras

np.random.seed(1)
tf.random.set_seed(1)


class Actor:
    def __init__(self,n_actions,n_features,learning_rate=1e-3):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate

        self.model = self._build_net()

    def _build_net(self):
        tf_obs = tf.keras.layers.Input(shape=(self.n_features,))

        # fc1
        layer = keras.layers.Dense(
            units=20,
            activation=keras.activations.relu,
            kernel_initializer=tf.random_normal_initializer(0., .1),
            bias_initializer=tf.constant_initializer(0.1),
        )(tf_obs)

        # fc2
        all_act_prob = keras.layers.Dense(
            units=self.n_actions,
            activation=keras.activations.softmax,
            kernel_initializer=tf.random_normal_initializer(0., .1),
            bias_initializer=tf.constant_initializer(0.1),
        )(layer)

        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr)
        model = keras.models.Model(inputs=tf_obs, outputs=all_act_prob)

        model.compile(loss='sparse_categorical_crossentropy', optimizer=self.optimizer)

        return model

    def choose_action(self, obs):
        prob_weights = self.model(obs)[0].numpy()
        action = np.random.choice(len(prob_weights), p=prob_weights)
        return action

    def learn(self, ob_s, a_s, r_s):
        self.model.fit(ob_s, np.array([a_s]), verbose=0, sample_weight=r_s)


class Critic:
    def __init__(self, n_features, learning_rate=1e-2, gama=.9):
        self.n_features = n_features
        self.lr = learning_rate
        self.gama = gama

        self.model = self._build_net()

    def _build_net(self):
        tf_obs = tf.keras.layers.Input(shape=(self.n_features,))

        # fc1
        layer = keras.layers.Dense(
            units=20,
            activation=keras.activations.relu,
            kernel_initializer=tf.random_normal_initializer(0., .1),
            bias_initializer=tf.constant_initializer(0.1),
        )(tf_obs)

        # fc2
        v_s = keras.layers.Dense(
            units=1,
            kernel_initializer=tf.random_normal_initializer(0., .1),
            bias_initializer=tf.constant_initializer(0.1),
        )(layer)

        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr)
        model = keras.models.Model(inputs=tf_obs, outputs=v_s)

        model.compile(loss='mse', optimizer=self.optimizer)

        return model

    def learn(self, ob_s, r_s, ob_s_):
        v_s_ = r_s + self.model(ob_s_) * self.gama
        e=v_s_-self.model(ob_s)
        self.model.fit(ob_s, v_s_, verbose=0)
        return e


DISPLAY_REWARD_THRESHOLD = 2e2  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time

env = gym.make('CartPole-v0')
env.seed(1)  # reproducible, general Policy gradient has high variance
env = env.unwrapped

N_F = env.observation_space.shape[0]
N_A = env.action_space.n
MAX_EP_STEPS = 1000  # maximum time step in one episode

actor = Actor(n_features=N_F, n_actions=N_A)
critic = Critic(n_features=N_F)

for i_episode in range(3000):
    s = env.reset()
    s = s[np.newaxis, :]
    t = 0
    track_r = []
    while True:
        if RENDER: env.render()

        a = actor.choose_action(s)
        s_, r, done, _ = env.step(a)

        s_ = s_[np.newaxis, :]

        if done: r = -20
        track_r.append(r)
        td_error = critic.learn(s, r, s_)
        actor.learn(s, a, td_error)  # true_gradient = grad[logPi(s,a) * td_error]

        s = s_
        t += 1

        if done or t >= MAX_EP_STEPS:
            ep_rs_sum = sum(track_r)
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
            print("episode:", i_episode, "  /reward:", int(running_reward))
            break
