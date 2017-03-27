
import random, util
from const import Const
from tanks import Env
from featureExtractors import SimpleExtractor
from featureExtractors import DangerExtractorInstance
import numpy as np
import tensorflow as tf




class DeepNet:
    def weight_variable(self, name, shape):
        return tf.get_variable(name, shape=shape,
                            initializer=tf.contrib.layers.xavier_initializer())
    def bias_variable(self, shape):
        return tf.Variable(tf.constant(0.1, shape=shape))

    def conv2d(self, x, W, strides):
        return tf.nn.conv2d(x, W, strides=strides, padding='VALID')

    def forward(self, x):
        W_conv1 = self.weight_variable('W_conv1', [8, 8, 1, 16])
        b_conv1 = self.bias_variable([16])
        h_conv1 = tf.nn.relu(self.conv2d(x, W_conv1, [1,4,4,1]) + b_conv1)    # stride 4

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

        #input size 25 -> stride 2, 4*4 -> output size
        W_fc1 = self.weight_variable('W_fc1', [12*12*32, 256])
        b_fc1 = self.bias_variable([256])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv2, W_fc1) + b_fc1)

        # 6 legal actions
        W_fc2 = self.weight_variable('W_fc2', [256, 6])
        b_fc2 = self.bias_variable([6])
        y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2


    def __init__(self):
        pass





class LinearQAgent:
    def __init__(self, epsilon=0.1, alpha=0.5, gamma=0.9, discount=0.9):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.discount = discount
        self.legalActions = Const.ACTIONS
        self.featExtractor = DangerExtractorInstance()
        self.featExtractor = SimpleExtractor()
        self.weights = [0] * self.featExtractor.featureNum
        self.game_state = None

        self.train_episodes = 10000

        self.env = Env()

    def getQValue(self, state, action):
        qvalue = 0.0
        for idx, value in enumerate(self.featExtractor.getFeatures(state, action)):
            qvalue += value * self.weights[idx]
        return qvalue

    def getMaxQValue(self, state):
        max_next_qvalue = None
        for nextAction in self.legalActions:
            next_qvalue = self.getQValue(state, nextAction)
            if max_next_qvalue is None or max_next_qvalue < next_qvalue:
                max_next_qvalue = next_qvalue
        if max_next_qvalue is None:
            max_next_qvalue = 0.0

        return max_next_qvalue

    def computeActionFromQValues(self, state):
        max_qvalue = None
        for action in self.legalActions:
            qvalue = self.getQValue(state, action)
            if max_qvalue is None or max_qvalue < qvalue:
                max_qvalue = qvalue

        if max_qvalue is None:
            return None

        actions = []
        for action in self.legalActions:
            qvalue = self.getQValue(state, action)
            if qvalue == max_qvalue:
                actions.append(action)

        if max_qvalue is not None and len(actions) == 0:
            return self.legalActions[0]
        if len(actions) > 1:
            return Const.DO_NOTHING
        return random.choice(actions)

    def getAction(self, state):
        if util.flipCoin(self.epsilon) is True:
            return random.choice(self.legalActions)
        return self.computeActionFromQValues(state)

    ''' @return reward, episode_over '''
    def step(self):
        if self.game_state is None:
            self.game_state, reward, episode_over, _ = self.env.step(0)
        else:
            action = self.getAction(self.game_state)
            next_game_state, reward, episode_over, _ = self.env.step(action)
            diff = reward + self.discount * self.getMaxQValue(next_game_state) - self.getQValue(self.game_state, action)
            for idx, feature_value in enumerate(self.featExtractor.getFeatures(self.game_state, action)):
                self.weights[idx] += self.alpha * diff * feature_value
            self.game_state = next_game_state
        return reward, episode_over

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


