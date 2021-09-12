from tensorflow import keras
import numpy as np, cv2, sys, tensorflow as tf

sys.path.append("game/")
import dqn_flappy_bird.game.wrapped_flappy_bird as game


class Policy_Gradient:
    def __init__(self):
        self.n_actions = 2  # number of valid actions

        # 4帧图像作输入，可以提取速度方向的信息
        self.n_features = 80 * 80 * 4  # number of valid actions

        self.gamma = 0.95
        self.gamma = 0.6
        self.lr = 1e-2
        self.max_r = 0

        self.ModelFP = 'models/model_diy.h5'

        # 每 save_model_step 步保存一次模型
        self.save_model_episode = 1e2
        self.episode = 0

        self.step = 0
        self.modify_lr_step = 1e2
        self.min_lr = 1e-4
        # 各层的初始权重
        self.kernel_initializer = 'truncated_normal'

        self.model = self._build_net((80, 80, 4))
        self.load_model()
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        self.ep_actions = []

    def _build_net(self, ob_shape):
        ob_s = tf.keras.layers.Input(shape=ob_shape)
        # 卷积1
        output1 = keras.layers.Conv2D(filters=32, kernel_size=(8, 8), kernel_initializer=self.kernel_initializer,
                                      activation="relu", padding="SAME")(ob_s)
        # 池化
        output1 = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same', data_format='channels_last')(output1)
        # 卷积2
        output1 = keras.layers.Conv2D(filters=64, kernel_size=(4, 4), activation="relu",
                                      kernel_initializer=self.kernel_initializer, padding="SAME")(output1)

        # 卷积3
        output1 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu",
                                      kernel_initializer=self.kernel_initializer, padding="SAME")(output1)
        # 拉平
        output1 = keras.layers.Flatten()(output1)

        # fc1
        output1 = keras.layers.Dense(512, kernel_initializer=self.kernel_initializer, activation='relu')(output1)
        # fc2
        # q = keras.layers.Dense(self.n_actions, kernel_initializer=self.kernel_initializer, activation='softmax')(output1)
        q = keras.layers.Dense(self.n_actions, kernel_initializer=self.kernel_initializer)(output1)
        q = keras.layers.BatchNormalization()(q)
        q = keras.layers.Activation('softmax')(q)

        model = keras.models.Model(inputs=ob_s, outputs=q)

        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr)
        # self.optimizer = keras.optimizers.RMSprop(learning_rate=self.lr)
        model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=[self.get_lr])

        return model

    def get_lr(self, y_true, y_pre):
        return self.optimizer.lr

    def preprocess_pic(self, pic):
        x_t = cv2.cvtColor(cv2.resize(pic, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
        return x_t / 255  # 图片做归一化

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def choose_action(self, s_t):
        s_t = s_t[np.newaxis, :]
        actions = self.model(s_t)[0].numpy()
        self.ep_actions.append(actions)
        action = np.random.choice(actions.shape[0], p=actions)

        return tf.one_hot(action, self.n_actions)

    def _discount_and_norm_rewards(self):
        if len(self.ep_rs) > self.max_r: self.max_r = len(self.ep_rs)
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        # discounted_ep_rs -= np.mean(discounted_ep_rs)
        # discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        for i in range(len(self.ep_rs)):
            print('/step {} \t/reward {} \t/discounted_ep_rs_norm {} \t/q {}\t/action {} '
                  .format(i, self.ep_rs[i], discounted_ep_rs_norm[i], self.ep_actions[i], np.argmax(self.ep_as[i])))

        obs_ = np.stack(self.ep_obs)
        as_ = np.stack(self.ep_as)

        # as_ = np.stack(self.ep_actions)*as_
        self.model.fit(obs_, as_, sample_weight=discounted_ep_rs_norm, verbose=2)

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []  # empty episode data
        self.ep_actions = []
        return discounted_ep_rs_norm

    def save_model(self):
        self.model.save_weights(self.ModelFP)
        print('>' * 88, 'model saved')

    def load_model(self):
        try:
            self.model.load_weights(self.ModelFP)
            print('加载成功')
        except Exception as e:
            print('没找到模型，不加载', e)

    def train(self):
        game_state = game.GameState()

        do_nothing = np.zeros(self.n_actions)
        do_nothing[0] = 1

        # image,reward,terminal
        x_t, _, _ = game_state.frame_step(do_nothing)
        x_t = self.preprocess_pic(x_t)
        s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

        while 1:
            self.step += 1
            action = self.choose_action(s_t)
            x_t_, reward, terminal = game_state.frame_step(action)

            s_t_ = self.preprocess_pic(x_t_)
            s_t_ = s_t_.reshape((80, 80, 1))
            s_t_ = np.append(s_t_, s_t[:, :, :3], axis=2)

            # 初始阶段跑到顶部的记忆太多，加大惩罚，加速训练(也许不需要？)
            x1 = np.where(s_t_[16, :63, 0] == 1)[0]
            # if len(x1) <= 2 and x1[0] < 10: reward -= 0.5
            if len(x1) <= 3 and x1[0] < 10: reward -= 0.5

            self.store_transition(s_t, action, reward)

            if terminal:
                self.step = 0
                self.episode += 1
                print('episode ', self.episode, ' \tmax_r ', self.max_r)
                self.learn()
                if self.episode % self.save_model_episode == self.save_model_episode - 1:
                    # pass
                    self.save_model()
                if self.episode % self.modify_lr_step == self.modify_lr_step - 1 and self.optimizer.lr > self.min_lr:
                    pass
                    # self.optimizer.lr = self.optimizer.lr * 0.5

            s_t = s_t_


dqn = Policy_Gradient()
dqn.train()
