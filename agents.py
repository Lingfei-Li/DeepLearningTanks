#!/usr/bin/python
# coding=utf-8

import os, pygame, time, random, uuid, sys, util
from optparse import OptionParser
from featureExtractors import SimpleExtractor
from const import Const
import matplotlib.pyplot as plt



class Agent:
    """
    An agent must define a getAction method, but may also define the
    following methods which will be called if they exist:

    def registerInitialState(self, state): # inspects the starting state
    """
    def __init__(self, index=0):
        self.index = index

    def getAction(self, state):
        """
        The Agent will receive a GameState and
        must return an action from Directions.{Fire, North, East, South, West}
        """
        util.raiseNotDefined()

class RandomAgent(Agent):
    def __init__(self, index=0):
        self.index = index
        self.last_move = 1
        self.epsilon = 0.9

    def getAction(self, state):
        """
        The Agent will receive a GameState and
        must return an action from Directions.{Fire, North, East, South, West}
        """
        print(state.map)
        for tile in state.map:
            print(tile.topleft)
            print(tile.width)
            print(tile.height)
            print(tile.type)

        # left, top, width, height, type

        if util.flipCoin(self.epsilon):
            return self.last_move
        move = random.choice([1,2,3,4])
        self.last_move = move
        return move

class ManualAgent(Agent):
    def __init__(self, index=0):
        self.index = index

    def getAction(self, state):
        """
        The Agent will receive a GameState and
        must return an action from Directions.{Fire, North, East, South, West}
        """
        util.raiseNotDefined()

class ValueEstimationAgent(Agent):
    """
      Abstract agent which assigns values to (state,action)
      Q-Values for an environment. As well as a value to a
      state and a policy given respectively by,

      V(s) = max_{a in actions} Q(s,a)
      policy(s) = arg_max_{a in actions} Q(s,a)

      Both ValueIterationAgent and QLearningAgent inherit
      from this agent. While a ValueIterationAgent has
      a model of the environment via a MarkovDecisionProcess
      (see mdp.py) that is used to estimate Q-Values before
      ever actually acting, the QLearningAgent estimates
      Q-Values while acting in the environment.
    """

    def __init__(self, alpha=1.0, epsilon=0.05, gamma=0.8, numTraining=10):
        """
        Sets options, which can be passed in via the Pacman command line using -a alpha=0.5,...
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.discount = float(gamma)
        self.numTraining = int(numTraining)

    ####################################
    #    Override These Functions      #
    ####################################
    def getQValue(self, state, action):
        """
        Should return Q(state,action)
        """
        util.raiseNotDefined()

    def getValue(self, state):
        """
        What is the value of this state under the best action?
        Concretely, this is given by

        V(s) = max_{a in actions} Q(s,a)
        """
        util.raiseNotDefined()

    def getPolicy(self, state):
        """
        What is the best action to take in the state. Note that because
        we might want to explore, this might not coincide with getAction
        Concretely, this is given by

        policy(s) = arg_max_{a in actions} Q(s,a)

        If many actions achieve the maximal Q-value,
        it doesn't matter which is selected.
        """
        util.raiseNotDefined()

    def getAction(self, state):
        """
        state: can call state.getLegalActions()
        Choose an action and return it.
        """
        util.raiseNotDefined()

class ReinforcementAgent(ValueEstimationAgent):
    """
      Abstract Reinforcemnt Agent: A ValueEstimationAgent
            which estimates Q-Values (as well as policies) from experience
            rather than a model

        What you need to know:
                    - The environment will call
                      observeTransition(state,action,nextState,deltaReward),
                      which will call update(state, action, nextState, deltaReward)
                      which you should override.
        - Use self.getLegalActions(state) to know which actions
                      are available in a state
    """

    ####################################
    #    Override These Functions      #
    ####################################

    def update(self, state, action, nextState, reward):
        """
                This class will call this function, which you write, after
                observing a transition and reward
        """
        util.raiseNotDefined()

    ####################################
    #    Read These Functions          #
    ####################################

    def getLegalActions(self, state):
        """
          Get the actions available for a given
          state. This is what you should use to
          obtain legal actions for a state
        """
        return self.actionFn(state)

    def observeTransition(self, state, action, nextState, deltaReward):
        """
            Called by environment to inform agent that a transition has
            been observed. This will result in a call to self.update
            on the same arguments

            NOTE: Do *not* override or call this function
        """
        self.episodeRewards += deltaReward
        self.update(state, action, nextState, deltaReward)


    def startEpisode(self):
        """
          Called by environment when new episode is starting
        """
        self.lastState = None
        self.lastAction = None
        self.episodeRewards = 0.0

        print("Agent Start Episode #" + str(self.episodesSoFar+1))

    def stopEpisode(self):
        """
          Called by environment when episode is done
        """
        if self.episodesSoFar < self.numTraining:
            self.accumTrainRewards += self.episodeRewards
        else:
            self.accumTestRewards += self.episodeRewards
        self.episodesSoFar += 1
        if self.episodesSoFar >= self.numTraining:
            # Take off the training wheels
            self.epsilon = 0.0  # no exploration
            self.alpha = 0.0  # no learning

        print("Agent Stop Episode")
        print(self.episodeRewards)
        for feature, weight in self.weights.iteritems():
            print("\t" + str(feature) + " - " + str(weight))
        self.episodeRewardsList.append(self.episodeRewards)

    def isInTraining(self):
        return self.episodesSoFar < self.numTraining

    def isInTesting(self):
        return not self.isInTraining()

    def __init__(self, actionFn=None, numTraining=100, epsilon=0.1, alpha=0.5, gamma=0.9):
        """
        actionFn: Function which takes a state and returns the list of legal actions

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        if actionFn == None:
            actionFn = lambda state: state.getLegalActions()
        self.actionFn = actionFn
        self.episodesSoFar = 0
        self.accumTrainRewards = 0.0
        self.accumTestRewards = 0.0
        self.numTraining = int(numTraining)
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.discount = float(gamma)
        self.episodeRewardsList = []

    ################################
    # Controls needed for Crawler  #
    ################################
    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def setLearningRate(self, alpha):
        self.alpha = alpha

    def setDiscount(self, discount):
        self.discount = discount

    def doAction(self, state, action):
        """
            Called by inherited class when
            an action is taken in a state
        """
        self.lastState = state
        self.lastAction = action


class PolicyGradientAgent(ReinforcementAgent):
    ''' Actor-Critic Agent '''
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self.weights = util.Counter()

        #100 episodes of training
        # self.weights['enemy'] = -10
        self.weights['bias'] = -10
        # self.weights['bullet'] = -200
        self.weights["horizontalEnemy"], self.weights["verticalEnemy"] = -200, -200
        self.weights["horizontalBullet"], self.weights["verticalBullet"] = -50, -50
        self.weights['edge'] = -10
        # self.weights['hitEnemy'] = 10
        # self.weights['moveFoward'] = 10

        self.legalActions = Const.ACTIONS
        self.featExtractor = SimpleExtractor()
        self.lastAction = 1

        
    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        qvalue = 0.0
        for feature_name, value in self.featExtractor.getFeatures(state, action).iteritems():
            qvalue += value * self.weights[feature_name]
        return qvalue


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        max_next_qvalue = None
        for nextAction in self.legalActions:
            next_qvalue = self.getQValue(state, nextAction)
            if max_next_qvalue is None or max_next_qvalue < next_qvalue:
                max_next_qvalue = next_qvalue
        if max_next_qvalue is None:
            max_next_qvalue = 0.0

        return max_next_qvalue

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"

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

    def getAction(self, state, discState):
        features = numpy.zeros(self.weights.shape[:-1])
        features[discState, :] = self.basis.computeFeatures(state)
        policy = self.getPolicy(features)
        return numpy.where(policy.cumsum() >= numpy.random.random())[0][0]

        
    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        "*** YOUR CODE HERE ***"
        # Epsilon greedy
        if util.flipCoin(self.epsilon) is True:
            self.lastAction = random.choice(self.legalActions)
        else:
            self.lastAction = self.computeActionFromQValues(state)
        return self.lastAction

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        diff = reward + self.discount * self.computeValueFromQValues(nextState) - self.getQValue(state, action)
        for feature_name, feature_value in self.featExtractor.getFeatures(state, action).iteritems():
            self.weights[feature_name] += self.alpha * diff * feature_value

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

    def final(self, state):
        print("Training Done")
        print("Total episodes: " + str(self.episodesSoFar))
        f = open('train_weight.txt', 'w')
        for feature, weight in self.weights.iteritems():
            f.write(str(feature) + " " + str(weight) + "\n" )
        f.close()  # you can omit in most cases as the destructor will call it

        plt.plot(self.episodeRewardsList)
        plt.ylabel("Episode Reward")
        plt.show()




class QLearningAgent(ReinforcementAgent):
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self.weights = util.Counter()

        #100 episodes of training
        self.weights['enemy'] = -10
        self.weights['bias'] = -10
        self.weights['bullet'] = -200
        self.weights['edge'] = -10
        self.weights['hitEnemy'] = 10
        self.weights['moveFoward'] = 10

        self.legalActions = Const.ACTIONS
        self.featExtractor = SimpleExtractor()
        self.lastAction = 1

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        qvalue = 0.0
        for feature_name, value in self.featExtractor.getFeatures(state, action).iteritems():
            qvalue += value * self.weights[feature_name]
        return qvalue

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        max_next_qvalue = None
        for nextAction in self.legalActions:
            next_qvalue = self.getQValue(state, nextAction)
            if max_next_qvalue is None or max_next_qvalue < next_qvalue:
                max_next_qvalue = next_qvalue
        if max_next_qvalue is None:
            max_next_qvalue = 0.0

        return max_next_qvalue

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"

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
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        "*** YOUR CODE HERE ***"
        # Epsilon greedy
        if util.flipCoin(self.epsilon) is True:
            self.lastAction = random.choice(self.legalActions)
        else:
            self.lastAction = self.computeActionFromQValues(state)
        return self.lastAction

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        diff = reward + self.discount * self.computeValueFromQValues(nextState) - self.getQValue(state, action)
        for feature_name, feature_value in self.featExtractor.getFeatures(state, action).iteritems():
            self.weights[feature_name] += self.alpha * diff * feature_value

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

    def final(self, state):
        print("Training Done")
        print("Total episodes: " + str(self.episodesSoFar))
        f = open('train_weight.txt', 'w')
        for feature, weight in self.weights.iteritems():
            f.write(str(feature) + " " + str(weight) + "\n" )
        f.close()  # you can omit in most cases as the destructor will call it

        plt.plot(self.episodeRewardsList)
        plt.ylabel("Episode Reward")
        plt.show()



