from tensorflow import keras
import numpy as np, time, cv2, sys, tensorflow as tf, math

sys.path.append("game/")
import dqn_flappy_bird.game.wrapped_flappy_bird as game


# train_on_batch + tensorboard 用
def named_logs(model, logs):
    result = {}
    for l in zip(model.metrics_names, logs):
        result[l[0]] = l[1]
    return result


class Dueling_DQN:
    GAME = 'bird'  # the name of the game being played for log files
    n_actions = 2  # number of valid actions

    # 4帧图像作输入，可以提取速度方向的信息
    n_features = 80 * 80 * 4  # number of valid actions

    gamma = 0.99  # decay rate of past observations
    OBSERVE = 2.
    EXPLORE = 10000.  # frames over which to anneal epsilon
    FINAL_EPSILON = 0.0001  # final value of epsilon
    INITIAL_EPSILON = 1  # 一开始随机探索
    # INITIAL_EPSILON = 0.0001  # 探索完后去掉随机
    epsilon = INITIAL_EPSILON

    # 一开始每5步choose_action，剩下4步不动，以更快更多的拿到reward=1的记录
    FRAME_PER_ACTION = 5
    # FRAME_PER_ACTION = 1
    lr = 1e-4

    batch_size = 32  # size of batch
    memory_size = 7000  # number of previous transitions to remember
    memory_count = 0

    # 初期reward=1概率极低，单独保存成功的记忆不被覆盖
    batch_plus_size = 8  # size of success batch
    memory_plus_size = 1000
    memory_plus_count = 0

    # 预计失败的记忆能学到更多东西，保证每次学习都有失败的记忆(成功的记忆一样)
    batch_minus_size = 8
    memory_minus_size = 1000
    memory_minus_count = 0
    update_model_step = 1e3

    ModelFP = 'models/model_diy.h5'

    # 每 save_model_step 步保存一次模型
    save_model_step = 2e3

    # 每 learn_step 步学习一次，也许不需要此参数
    learn_step = 5

    step = 0

    # 各层的初始权重
    kernel_initializer = 'truncated_normal'

    # cmd里 tensorboard --logdir=**/logs，可以localhost:6006查看训练曲线
    # 有时logdir要输入全路径，否则找不到训练数据
    tensorboard = keras.callbacks.TensorBoard(
        log_dir='logs',  # TensorBoard文件保存的路径
        batch_size=batch_size,
    )

    def __init__(self):
        self.memory = np.zeros((self.memory_size, 2 + 2 * self.n_features + self.n_actions))
        self.memory_plus = np.zeros((self.memory_plus_size, 2 + 2 * self.n_features + self.n_actions))
        self.memory_minus = np.zeros((self.memory_minus_size, 2 + 2 * self.n_features + self.n_actions))
        self.model = self._build_net((80, 80, 4))
        self.model_target = self._build_net((80, 80, 4))
        self.tensorboard.set_model(self.model)
        self.load_model()

    def _build_net(self, input_shape):

        input1 = tf.keras.layers.Input(shape=input_shape)
        # 卷积1
        output1 = keras.layers.Conv2D(filters=32, kernel_size=(8, 8), kernel_initializer=self.kernel_initializer,
                                      activation="relu", padding="SAME")(input1)
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
        # Advantage fc
        advantage = keras.layers.Dense(self.n_actions, kernel_initializer=self.kernel_initializer)(output1)
        # Value fc
        value = keras.layers.Dense(1, kernel_initializer=self.kernel_initializer)(output1)
        # Q fc
        # fc=keras.layers.Add()([fc2, fc3])         #加法也能用这个，但不知道怎么减去reduce_mean
        q = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))

        model = keras.models.Model(inputs=input1, outputs=q)

        optimizer = keras.optimizers.Adam(learning_rate=self.lr)
        model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])

        return model

    def preprocess_pic(self, pic):
        x_t = cv2.cvtColor(cv2.resize(pic, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)

        # x=np.where(x_t[:,:63]==255)
        # print("x",x)
        # cv2.imshow('x', x_t);
        # cv2.waitKey(0)

        return x_t / 255  # 图片做归一化

    # 分开保存成功/失败/其他的记忆(也许不需要分开)
    def store_transition(self, s_t, action, reward, terminal, s_t_):
        data = np.hstack((s_t.flatten(), action, reward, terminal, s_t_.flatten()))
        if reward == 1:
            index = self.memory_plus_count % self.memory_plus_size
            self.memory_plus[index] = data
            self.memory_plus_count += 1
        elif reward <= -1:
            index = self.memory_minus_count % self.memory_minus_size
            self.memory_minus[index] = data
            self.memory_minus_count += 1
        else:
            index = self.memory_count % self.memory_size
            self.memory[index] = data
            self.memory_count += 1

    def choose_action(self, s_t):
        if self.step % self.FRAME_PER_ACTION:
            action = 0
        else:
            if np.random.uniform() > self.epsilon:
                s_t = s_t[np.newaxis, :]
                actions_value = self.model.predict(s_t)
                # action=np.random.choice(np.where(actions_value == np.max(actions_value))[1])
                action = np.argmax(actions_value, axis=1)[0]
            else:
                action = np.random.randint(0, self.n_actions)
        return tf.one_hot(action, self.n_actions)

    # 分开获取记忆，合并后打乱
    def choose_batch_data(self):
        sample_index = np.random.choice(min(self.memory_count, self.memory_size),
                                        self.batch_size - self.batch_plus_size - self.batch_minus_size)
        sample_plus_index = np.random.choice(min(self.memory_plus_count, self.memory_plus_size), self.batch_plus_size)
        sample_minus_index = np.random.choice(min(self.memory_minus_count, self.memory_minus_size),
                                              self.batch_minus_size)
        x1, x2, x3 = self.memory[sample_index], self.memory_plus[sample_plus_index], self.memory_minus[
            sample_minus_index]
        batch = np.vstack((x1, x2, x3))
        np.random.shuffle(batch)
        o_s = batch[:, :self.n_features].reshape(-1, 80, 80, 4)
        a_s = batch[:, self.n_features:self.n_features + self.n_actions]
        a_s = np.argmax(a_s, axis=1)
        r_s = batch[:, self.n_features + self.n_actions]
        t_s = batch[:, 1 + self.n_features + self.n_actions]
        o_s_ = batch[:, -self.n_features:].reshape(-1, 80, 80, 4)
        return o_s, a_s, r_s, t_s, o_s_

    def learn(self):
        if self.step % self.update_model_step == 0:
            self.model_target.set_weights(self.model.get_weights())

        o_s, a_s, r_s, t_s, o_s_ = self.choose_batch_data()

        q_eval = self.model.predict(o_s, batch_size=self.batch_size)
        q_next_target = self.model_target.predict(o_s_, batch_size=self.batch_size)

        q_target = q_eval.copy()

        target_part = r_s + (1 - t_s) * self.gamma * np.max(q_next_target, axis=1)

        q_target[range(self.batch_size), a_s] = target_part
        history = self.model.train_on_batch(o_s, q_target)

        # train_on_batch + tensorboard 用
        self.tensorboard.on_epoch_end(self.step, named_logs(self.model, history))

        if self.epsilon > self.FINAL_EPSILON:
            self.epsilon -= (self.INITIAL_EPSILON - self.FINAL_EPSILON) / self.EXPLORE

    def save_model(self):
        self.model.save_weights(self.ModelFP)
        print('>' * 88, 'model saved')

    def load_model(self):
        try:
            self.model.load_weights(self.ModelFP)
            self.model_target.load_weights(self.ModelFP)
        except Exception:
            print('没找到模型，不加载')

    def train(self):
        game_state = game.GameState()

        do_nothing = np.zeros(self.n_actions)
        do_nothing[0] = 1

        # image,reward,terminal
        x_t, _, _ = game_state.frame_step(do_nothing)
        x_t = self.preprocess_pic(x_t)
        s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

        while 1:
            action = self.choose_action(s_t)
            x_t_, reward, terminal = game_state.frame_step(action)
            # RL take action and get next observation and reward

            s_t_ = self.preprocess_pic(x_t_)
            s_t_ = s_t_.reshape((80, 80, 1))
            s_t_ = np.append(s_t_, s_t[:, :, :3], axis=2)
            # s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

            # 初始阶段跑到顶部的记忆太多，加大惩罚，加速训练(也许不需要？)
            x1 = np.where(s_t_[16, :63, 0] == 1)[0]
            if len(x1) <= 2 and x1[0] < 10: reward -= 0.5
            # print(x1)

            self.store_transition(s_t, action, reward, terminal, s_t_)

            if (self.memory_plus_count >= self.OBSERVE) and (self.step % self.learn_step == 0):
                self.FRAME_PER_ACTION = max(1, math.ceil(self.FRAME_PER_ACTION * (1 - self.step / self.EXPLORE)))
                self.INITIAL_EPSILON = .5
                if self.epsilon > .5: self.epsilon = .5
                self.learn()

            s_t = s_t_

            self.step += 1

            if self.memory_plus_count < self.OBSERVE:
                state = "observe"
            elif self.memory_plus_count >= self.OBSERVE and self.step <= self.EXPLORE:
                state = "explore"
            else:
                state = "train"

            print("TIMESTEP", self.step, "/ STATE", state, "/ EPSILON", self.epsilon, "/ REWARD", reward)

            if self.epsilon > self.FINAL_EPSILON and self.memory_plus_count > self.OBSERVE:
                self.epsilon -= (self.INITIAL_EPSILON - self.FINAL_EPSILON) / self.EXPLORE

            if self.step % self.save_model_step == self.save_model_step - 1:
                self.save_model()


dqn = Dueling_DQN()
dqn.train()
