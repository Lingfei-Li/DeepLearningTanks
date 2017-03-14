
import random, util
from const import Const
from tanks import Env
from featureExtractors import SimpleExtractor, PositionExtractor
from linearQAgent import LinearQAgent
import numpy as np
import tensorflow as tf


# noinspection PyTypeChecker
class ANNAgent:
    def __init__(self, epsilon=0.1, alpha=0.5, gamma=0.9, discount=0.9):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.discount = discount
        self.legalActions = Const.ACTIONS
        self.featExtractor = SimpleExtractor()
        self.featExtractor2 = PositionExtractor()
        self.weights = util.Counter()
        self.game_state = None

        self.train_episodes = 10000
        self.level_type = "minimal"
        self.game_speed = 1000

        self.env = Env(self.level_type, self.game_speed, self.train_episodes)

        self.input_num = self.featExtractor2.getFeatureNum()
        self.hidden_num = 100
        self.output_num = 6
        self.W1 = np.random.rand(self.input_num, self.hidden_num)
        self.W2 = np.random.rand(self.hidden_num, self.output_num)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]

    def forward_prop(self, input):
        hidden_net = np.dot(input, self.W1)

        hidden_out = self.sigmoid(hidden_net)

        output_net = np.dot(hidden_out, self.W2)

        output_out = self.sigmoid(output_net)

        return output_out

    def start_episode(self, episode_num):
        self.env.reset()
        total_reward = 0.0

        dlopgs, drewards = [], []
        while True:
            if self.game_state is None:
                self.game_state, reward, episode_over = self.env.step(0)
                continue

            #get action for the current state
            ann_output = self.forward_prop(self.game_state)

            action = np.argmax(ann_output)

            action_array = np.zeros(self.output_num)
            action_array[action] = 1

            dlopgs.append(action_array - ann_output)

            next_game_state, reward, episode_over = self.env.step(action)

            drewards.append(reward)



            self.game_state = next_game_state


            total_reward += reward
            if episode_over: break

        #episode over

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
    agent = LinearQAgent()
    # agent = ANNAgent()
    agent.start_train()



