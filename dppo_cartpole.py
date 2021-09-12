import tensorflow as tf
from tensorflow import keras
from keras.layers import *
import numpy as np
import gym, threading, queue

np.random.seed(1)
tf.random.set_seed(1)

EP_MAX = 1000
EP_LEN = 500
N_WORKER = 4                # parallel workers
GAMMA = 0.9                 # reward discount factor
A_LR = 0.0001               # learning rate for actor
C_LR = 0.0001               # learning rate for critic
MIN_BATCH_SIZE = 64         # minimum batch size for updating PPO
UPDATE_STEP = 15            # loop update operation n-steps
EPSILON = 0.2               # for clipping surrogate objective
GAME = 'CartPole-v0'

env = gym.make(GAME)
env.seed(1)
S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.n
print(S_DIM,A_DIM)

class PPO(object):
    def __init__(self):
        self.opt_a = tf.compat.v1.train.AdamOptimizer(A_LR)
        self.opt_c = tf.compat.v1.train.AdamOptimizer(C_LR)

        self.model_a = self._build_anet(trainable=True)
        self.model_a_old = self._build_anet(trainable=False)
        self.model_c = self._build_cnet()

    def _build_anet(self,trainable=True):
        tfs_a = Input([S_DIM], )
        l1 = Dense(200, 'relu',trainable=trainable)(tfs_a)
        a_prob = Dense(A_DIM, 'softmax',trainable=trainable)(l1)
        model_a = keras.models.Model(inputs=tfs_a, outputs=a_prob)
        return model_a

    def _build_cnet(self):
        tfs_c = Input([S_DIM], )
        l1 = Dense(200, 'relu')(tfs_c)
        v = Dense(1)(l1)
        model_c = keras.models.Model(inputs=tfs_c, outputs=v)
        model_c.compile(optimizer=self.opt_c, loss='mse')
        return model_c

    def update(self):
        global GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            if GLOBAL_EP < EP_MAX:
                UPDATE_EVENT.wait()                     # wait until get batch of data
                self.model_a_old.set_weights(self.model_a.get_weights())
                data = [QUEUE.get() for _ in range(QUEUE.qsize())]      # collect data from all workers
                data = np.vstack(data)
                s, a, r = data[:, :S_DIM], data[:, S_DIM: S_DIM + 1].ravel(), data[:, -1:]

                v = self.get_v(s)
                adv = r - v
                oldpi = self.model_a_old(s)
                for i in range(UPDATE_STEP):
                    with tf.GradientTape() as tape:
                        pi=self.model_a(s)
                        # xx=tf.shape(a)[0]
                        # xxx=tf.range(xx, dtype=tf.int32)
                        a_indices = tf.stack([tf.range(tf.shape(a)[0], dtype=tf.int32), a], axis=1)
                        pi_prob = tf.gather_nd(params=pi, indices=a_indices)
                        oldpi_prob = tf.gather_nd(params=oldpi, indices=a_indices)

                        ratio = pi_prob / (oldpi_prob + 1e-5)
                        surr = ratio * adv
                        x2 = tf.clip_by_value(ratio, 1. - EPSILON, 1. + EPSILON) * adv
                        x3 = tf.minimum(surr, x2)
                        aloss = -tf.reduce_mean(x3)

                    a_grads = tape.gradient(aloss, self.model_a.trainable_weights)
                    a_grads_and_vars = zip(a_grads, self.model_a.trainable_weights)
                    self.opt_a.apply_gradients(a_grads_and_vars)

                self.model_c.fit(s, r, verbose=1, shuffle=False,epochs=UPDATE_STEP)

                UPDATE_EVENT.clear()        # updating finished
                GLOBAL_UPDATE_COUNTER = 0   # reset counter
                ROLLING_EVENT.set()         # set roll-out available



    def choose_action(self, s):
        s = s[np.newaxis, :]
        prob_weights = self.model_a(s)[0].numpy()
        action = np.random.choice(len(prob_weights),p=prob_weights)
        return action

    def get_v(self, s):
        s=s.reshape(-1,4)
        v = self.model_c(s)
        return v[0,0]



class Worker(object):
    def __init__(self, wid):
        self.wid = wid
        self.env = gym.make(GAME).unwrapped
        self.ppo = GLOBAL_PPO

    def work(self):
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            s = self.env.reset()
            ep_r = 0
            buffer_s, buffer_a, buffer_r = [], [], []
            for t in range(EP_LEN):
                if not ROLLING_EVENT.is_set():                  # while global PPO is updating
                    ROLLING_EVENT.wait()                        # wait until PPO is updated
                    buffer_s, buffer_a, buffer_r = [], [], []   # clear history buffer, use new policy to collect data
                a = self.ppo.choose_action(s)
                s_, r, done, _ = self.env.step(a)
                if done: r = -10
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r-1)                            # 0 for not down, -11 for down. Reward engineering
                s = s_
                ep_r += r

                GLOBAL_UPDATE_COUNTER += 1                      # count to minimum batch size, no need to wait other workers
                if t == EP_LEN - 1 or GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE or done:
                    if done:
                        v_s_ = 0                                # end of episode
                    else:
                        v_s_ = self.ppo.get_v(s_)

                    discounted_r = []                           # compute discounted reward
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, None]
                    buffer_s, buffer_a, buffer_r = [], [], []
                    QUEUE.put(np.hstack((bs, ba, br)))          # put data in the queue
                    if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                        ROLLING_EVENT.clear()       # stop collecting data
                        UPDATE_EVENT.set()          # globalPPO update

                    if GLOBAL_EP >= EP_MAX:         # stop training
                        COORD.request_stop()
                        break

                    if done: break

            # record reward changes, plot later
            if len(GLOBAL_RUNNING_R) == 0: GLOBAL_RUNNING_R.append(ep_r)
            else: GLOBAL_RUNNING_R.append(GLOBAL_RUNNING_R[-1]*0.9+ep_r*0.1)
            GLOBAL_EP += 1
            print('{0:.1f}%'.format(GLOBAL_EP/EP_MAX*100), '|W%i' % self.wid,  '|Ep_r: %.2f' % ep_r,)


if __name__ == '__main__':
    GLOBAL_PPO = PPO()
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    UPDATE_EVENT.clear()            # not update now
    ROLLING_EVENT.set()             # start to roll out
    workers = [Worker(wid=i) for i in range(N_WORKER)]

    GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
    GLOBAL_RUNNING_R = []
    COORD = tf.train.Coordinator()
    QUEUE = queue.Queue()           # workers putting data in this queue
    threads = []
    for worker in workers:          # worker threads
        t = threading.Thread(target=worker.work, args=())
        t.start()                   # training
        threads.append(t)
    # add a PPO updating thread
    threads.append(threading.Thread(target=GLOBAL_PPO.update,))
    threads[-1].start()
    COORD.join(threads)

    env = gym.make('CartPole-v0')
    while True:
        s = env.reset()
        for t in range(1000):
            env.render()
            s, r, done, info = env.step(GLOBAL_PPO.choose_action(s))
            if done:
                break