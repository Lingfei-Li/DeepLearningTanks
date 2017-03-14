import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import matplotlib.pyplot as plt
from tanks import Env
from const import Const
from featureExtractors import DangerExtractor
from featureExtractors import SimpleExtractor


# env = gym.make('CartPole-v0')
train_episodes = 100
learning_rate = 0.01
level_type = "minimal"
game_speed = 1000
env = Env(level_type, game_speed, train_episodes)
gamma = 0.99
featExtractor = SimpleExtractor()

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class agent():
    def __init__(self, lr, s_size, a_size, h_size):
        # These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        self.state_in = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
        hidden = slim.fully_connected(self.state_in, h_size, biases_initializer=None, activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden, a_size, activation_fn=tf.nn.softmax, biases_initializer=None)
        self.chosen_action = tf.argmax(self.output, 1)

        # The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        # to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)

        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) * self.reward_holder)

        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx, var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss, tvars)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))


tf.reset_default_graph()  # Clear the Tensorflow graph.

myAgent = agent(lr=learning_rate, s_size=featExtractor.featureNum, a_size=5, h_size=50)  # Load the agent.

total_episodes = 5000  # Set total number of episodes to train agent on.
max_ep = 99999
update_frequency = 5

init = tf.initialize_all_variables()

# Launch the tensorflow graph
with tf.Session() as sess:
    sess.run(init)
    i = 0
    total_reward = []
    total_lenght = []

    gradBuffer = sess.run(tf.trainable_variables())
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    while i < total_episodes:
        s = env.reset()
        s = featExtractor.getFeatures(s, Const.DO_NOTHING)
        running_reward = 0
        ep_history = []
        # for j in range(max_ep):
        action_freq = np.zeros(len(Const.ACTIONS) + 1)
        while True:
            env.render()

            # Choose either a random action or one from our network.
            epsilon = 0.1
            if np.random.rand() < epsilon:
                a = np.random.random_integers(0, 4)
            elif np.random.rand() < 0.5 and len(ep_history) > 0:
                a = ep_history[len(ep_history)-1][1]
            else:
                a_dist = sess.run(myAgent.output, feed_dict={myAgent.state_in: [s]})
                a = np.random.choice(a_dist[0], p=a_dist[0])
                a = np.argmax(a_dist == a)
            action_freq[a] += 1

            s1, r, d, _ = env.step(a+1)  # Get our reward for taking an action given a bandit.
            s1 = featExtractor.getFeatures(s1, Const.DO_NOTHING)
            ep_history.append([s, a, r, s1])
            s = s1
            running_reward += r
            if d == True:
                # Update the network.
                ep_history = np.array(ep_history)
                ep_history[:, 2] = discount_rewards(ep_history[:, 2])
                feed_dict = {myAgent.reward_holder: ep_history[:, 2],
                             myAgent.action_holder: ep_history[:, 1], myAgent.state_in: np.vstack(ep_history[:, 0])}
                grads = sess.run(myAgent.gradients, feed_dict=feed_dict)
                for idx, grad in enumerate(grads):
                    gradBuffer[idx] += grad

                if i % update_frequency == 0 and i != 0:
                    feed_dict = dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
                    _ = sess.run(myAgent.update_batch, feed_dict=feed_dict)
                    for ix, grad in enumerate(gradBuffer):
                        gradBuffer[ix] = grad * 0

                total_reward.append(running_reward)
                # total_lenght.append(j)
                # print("Episode #", i, "Reward ", running_reward)
                break


                # Update our running tally of scores.
        if i % 10 == 0:
            print("Total: ", i, "Last 10: ", np.mean(total_reward[-10:]))
            total_actions = np.sum(action_freq)
            print("Action distribution (%): ")
            action_dist = ""
            for j in range(0, len(action_freq)):
                action_dist += str(int((action_freq[j]*100/total_actions))) + ' '
            print("\t", action_dist)
        i += 1
    print(total_reward)