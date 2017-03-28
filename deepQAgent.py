
import random, util
from const import Const
from tanks import Env
import numpy as np
import tensorflow as tf




class DeepNet:
    @staticmethod
    def weight_variable(name, shape):
        return tf.get_variable(name, shape=shape,
                            initializer=tf.contrib.layers.xavier_initializer())
    @staticmethod
    def bias_variable(shape):
        return tf.Variable(tf.constant(0.1, shape=shape))

    @staticmethod
    def conv2d(x, W, strides):
        return tf.nn.conv2d(x, W, strides=strides, padding='VALID')

    def __init__(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 104, 104, 1])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 6])

        W_conv1 = self.weight_variable('W_conv1', [8, 8, 1, 16])
        b_conv1 = self.bias_variable([16])
        h_conv1 = tf.nn.relu(self.conv2d(self.x, W_conv1, [1,4,4,1]) + b_conv1)    # stride 4

        #from 231n:
        #Accepts a volume of size W1×H1×D1W1×H1×D1
        # Requires four hyperparameters:
        # Number of filters K,
        # their spatial extent F,
        # the stride S,
        # the amount of zero padding P.
        # Produces a volume of size W2×H2×D2W2×H2×D2 where:
        # W2=(W1−F+2P)/S+1W2=(W1−F+2P)/S+1
        # H2=(H1−F+2P)/S+1H2=(H1−F+2P)/S+1 (i.e. width and height are computed equally by symmetry)
        # D2=K

        #input size 104 -> stride 4, 8*8 -> output size 25
        W_conv2 = self.weight_variable('W_conv2', [4, 4, 16, 32])
        b_conv2 = self.bias_variable([32])
        h_conv2 = tf.nn.relu(self.conv2d(h_conv1, W_conv2, [1,2,2,1]) + b_conv2)    # stride 2
        h_conv2_flat = tf.reshape(h_conv2, [-1, 11*11*32])

        #input size 25 -> stride 2, 4*4 -> output size
        W_fc1 = self.weight_variable('W_fc1', [11*11*32, 256])
        b_fc1 = self.bias_variable([256])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

        # 6 legal actions
        W_fc2 = self.weight_variable('W_fc2', [256, 6])
        b_fc2 = self.bias_variable([6])
        self.y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def forward(self, x):
        return self.sess.run(self.y_conv, feed_dict={self.x:x})

    def backward(self, x, y):
        self.train_step.run(feed_dict={self.x:x, self.y_:y})




class DeepQAgent:
    def __init__(self, epsilon=0.1, alpha=0.5, gamma=0.9, discount=0.9):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.discount = discount
        self.legalActions = Const.ACTIONS
        self.game_state = None

        self.nn = DeepNet()

        self.train_episodes = 10

        self.env = Env()


    ''' @return reward, episode_over '''
    def step(self):
        if self.game_state is None:
            self.game_state, reward, episode_over, _ = self.env.step(0)
        else:
            # get max-q-value action from nn
            # calculate diff
            # update q value with diff

            self.game_state = self.game_state.reshape(1, 104, 104, 1)


            action = 0
            next_game_state, reward, episode_over, _ = self.env.step(action)
            # self.nn.forward(self.game_state)
            # self.nn.backward(self.game_state, [[1,0,0,0,0,0]])

            # diff = reward + self.discount * self.getMaxQValue(next_game_state) - self.getQValue(self.game_state, action)
            # for idx, feature_value in enumerate(self.featExtractor.getFeatures(self.game_state, action)):
            #     self.weights[idx] += self.alpha * diff * feature_value
            # self.game_state = next_game_state
            pass
        # return reward, episode_over
        return 0, False

    def start_episode(self, episode_num):
        self.env.reset()
        total_reward = 0.0
        while True:
            self.env.render()
            reward, episode_over = self.step()
            total_reward += reward
            if episode_over: break
        return total_reward

    def start_train(self):
        episode_cnt = 0
        total_reward = []
        while episode_cnt < self.train_episodes:
            episode_cnt += 1
            total_reward.append(self.start_episode(episode_cnt))
            if episode_cnt % 10 == 0:
                print("Episode #" + str(episode_cnt))
                print("10 avg: ", np.mean(total_reward[-10:]))


if __name__ == "__main__":
    agent = DeepQAgent()
    agent.start_train()


