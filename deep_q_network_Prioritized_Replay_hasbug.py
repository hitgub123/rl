from tensorflow import keras
import numpy as np, time, cv2, sys, tensorflow as tf, math

sys.path.append("game/")
import dqn_flappy_bird.game.wrapped_flappy_bird as game

'''
    bug：自定义loss函数mse_with_weight里，weights=self.ISWeights,        
    建模型时估计就写死了self.ISWeights，不会发生变化，导致模型训练便慢
'''
# train_on_batch + tensorboard 用
def named_logs(model, logs):
    result = {}
    for l in zip(model.metrics_names, logs):
        result[l[0]] = l[1]
    return result


class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:  # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:  # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1  # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)  # set the max p for new p

    def sample(self, n):
        #这里以为ISWeights的shape np.empty((n, 1)可以写出np.empty((n,)，就改了。
        #结果发现计算loss时tf无法进行广播，又reshape了一次。相关的代码全部删除即可，画蛇添足了
        # b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n,))
        pri_seg = self.tree.total_p / n  # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p  # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            # ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            ISWeights[i] = np.power(prob / min_prob, -self.beta)
            # print(idx, p ,ISWeights[i])
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            # print(ti,p)
            self.tree.update(ti, p)


class Prioritized_Replay_DQN:
    GAME = 'bird'  # the name of the game being played for log files
    n_actions = 2  # number of valid actions

    # 4帧图像作输入，可以提取速度方向的信息
    n_features = 80 * 80 * 4  # number of valid actions

    gamma = 0.99  # decay rate of past observations
    OBSERVE = 1000.
    EXPLORE = 50000.  # frames over which to anneal epsilon
    FINAL_EPSILON = 0.0001  # final value of epsilon
    INITIAL_EPSILON = 1  # 一开始随机探索
    # INITIAL_EPSILON = 0.0001  # 探索完后去掉随机
    epsilon = INITIAL_EPSILON

    # 一开始每5步choose_action，剩下4步不动，以更快更多的拿到reward=1的记录
    FRAME_PER_ACTION = 5
    # FRAME_PER_ACTION=1
    lr = 1e-4

    batch_size = 32  # size of batch
    memory_size = 7000  # number of previous transitions to remember

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
        # batch_size is no longer needed in the `TensorBoard` Callback and will be ignored in TensorFlow 2.0.
        batch_size=batch_size,
    )

    def __init__(self):
        self.memory = Memory(capacity=self.memory_size)

        self.model = self._build_net((80, 80, 4))
        self.model_target = self._build_net((80, 80, 4))

        self.tensorboard.set_model(self.model)
        self.load_model()

    def _build_net(self, input_shape):
        model = keras.Sequential()

        conv1 = keras.layers.Conv2D(filters=32, input_shape=input_shape, kernel_size=(8, 8),
                                    kernel_initializer=self.kernel_initializer, activation="relu", padding="SAME")
        model.add(conv1)

        pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same', data_format='channels_last')
        model.add(pool1)

        conv2 = keras.layers.Conv2D(filters=64, kernel_size=(4, 4), activation="relu",
                                    kernel_initializer=self.kernel_initializer, padding="SAME")
        model.add(conv2)

        conv3 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu",
                                    kernel_initializer=self.kernel_initializer, padding="SAME")
        model.add(conv3)

        flat = keras.layers.Flatten()
        model.add(flat)

        fc1 = keras.layers.Dense(512, kernel_initializer=self.kernel_initializer, activation='relu')
        model.add(fc1)
        fc2 = keras.layers.Dense(self.n_actions, kernel_initializer=self.kernel_initializer)
        model.add(fc2)

        optimizer = keras.optimizers.Adam(learning_rate=self.lr)
        model.compile(loss=self.mse_with_weight, optimizer=optimizer, metrics=['mse'])

        return model

    def mse_with_weight(self, y_true, y_pred):
        # print(y_true.shape,y_pred.shape,self.ISWeights.shape)
        weights = tf.reshape(self.ISWeights, (-1, 1))
        weights = tf.cast(weights, y_true.dtype)
        loss = tf.reduce_mean(weights * tf.square(y_true - y_pred))
        return loss

    def preprocess_pic(self, pic):
        x_t = cv2.cvtColor(cv2.resize(pic, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
        return x_t / 255  # 图片做归一化

    def store_transition(self, s_t, action, reward, s_t_):
        data = np.hstack((s_t.flatten(), action, reward, s_t_.flatten()))
        self.memory.store(data)

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

    def learn(self):
        if self.step % self.update_model_step == 0:
            self.model_target.set_weights(self.model.get_weights())

        tree_idx, batch_memory, self.ISWeights = self.memory.sample(self.batch_size)
        o_s = batch_memory[:, :self.n_features].reshape(-1, 80, 80, 4)
        a_s = batch_memory[:, self.n_features:self.n_features + self.n_actions]
        a_s = np.argmax(a_s, axis=1)
        r_s = batch_memory[:, self.n_features + self.n_actions]
        o_s_ = batch_memory[:, -self.n_features:].reshape(-1, 80, 80, 4)

        q_next = self.model_target.predict(o_s_, batch_size=self.batch_size)
        q_eval = self.model.predict(o_s, batch_size=self.batch_size)

        q_target = q_eval.copy()
        target_part = r_s + self.gamma * np.max(q_next, axis=1)

        abs_errors = np.abs(q_target[range(self.batch_size), a_s] - target_part)
        self.memory.batch_update(tree_idx, abs_errors)

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

            # 初始阶段跑到顶部的记忆太多，加大惩罚，加速训练(也许不需要？)
            x1 = np.where(s_t_[16, :63, 0] == 1)[0]
            if len(x1) <= 2 and x1[0] < 10: reward -= 0.5

            self.store_transition(s_t, action, reward, s_t_)

            # if (self.step >= self.OBSERVE) and (self.step % self.learn_step == 0):
            if (self.step >= self.memory_size) and (self.step % self.learn_step == 0):
                self.FRAME_PER_ACTION = max(1, math.ceil(self.FRAME_PER_ACTION * (1 - self.step / self.EXPLORE)))
                self.INITIAL_EPSILON = .5
                if self.epsilon > .5: self.epsilon = .5
                self.learn()

            s_t = s_t_

            self.step += 1

            if self.step < self.OBSERVE:
                state = "observe"
            elif self.step >= self.OBSERVE and self.step <= self.EXPLORE:
                state = "explore"
            else:
                state = "train"

            print("TIMESTEP", self.step, "/ STATE", state, "/ EPSILON", self.epsilon, "/ REWARD", reward)

            if self.epsilon > self.FINAL_EPSILON and self.step > self.OBSERVE:
                self.epsilon -= (self.INITIAL_EPSILON - self.FINAL_EPSILON) / self.EXPLORE

            if self.step % self.save_model_step == self.save_model_step - 1:
                self.save_model()


dqn = Prioritized_Replay_DQN()
dqn.train()
