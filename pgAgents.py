#!/usr/bin/python
# coding=utf-8

import os, pygame, time, random, uuid, sys, util, math
from optparse import OptionParser
from featureExtractors import SimpleExtractor
from const import Const
import matplotlib.pyplot as plt
from agents import ReinforcementAgent


class PolicyGradientAgent(ReinforcementAgent):
    ''' Actor-Critic Agent '''
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self.q_value_weights = util.Counter()
        self.policy_weights = util.Counter()

        #learning rate for policy weights
        self.beta = 0.1

        self.legalActions = Const.ACTIONS
        self.featExtractor = SimpleExtractor()
        self.lastAction = 1

        
        #100 episodes of training
        self.q_value_weights['enemy'] = -10
        self.q_value_weights['bias'] = -10
        self.q_value_weights['bullet'] = -200
        self.q_value_weights['edge'] = -10
        self.q_value_weights['hitEnemy'] = 10
        self.q_value_weights['moveFoward'] = 10

        
    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        qvalue = 0.0
        for feature_name, value in self.featExtractor.getFeatures(state, action).iteritems():
            qvalue += value * self.q_value_weights[feature_name]
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
            return random.choice(self.legalActions)
        
        max_policy_value = None
        max_action = -1
        for action, value in self.softmaxPolicy(state).iteritems():
            if max_policy_value is None or max_policy_value < value:
                max_action = action
                max_policy_value = value
        return max_action
        
        
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
        
        #update q value weights (omega)
        diff = reward + self.discount * self.computeValueFromQValues(nextState) - self.getQValue(state, action)
        for feature_name, feature_value in self.featExtractor.getFeatures(state, action).iteritems():
            self.q_value_weights[feature_name] += self.alpha * diff * feature_value
            
        #update policy weights (theta)
        
        expectedFeatureValues = util.Counter()
        for action in self.legalActions:
            for feature_name, value in self.featExtractor.getFeatures(state, action).iteritems():
                expectedFeatureValues[feature_name] += value
        for feature_name, value in expectedFeatureValues.iteritems():
            expectedFeatureValues[feature_name] /= len(self.legalActions)
        
        for feature_name, value in self.featExtractor.getFeatures(state, action).iteritems():
            scoreFunc = value - expectedFeatureValues[feature_name]
            self.policy_weights[feature_name] += self.beta * scoreFunc * self.getQValue(state, action)
        
        
    def softmaxPolicy(self, state):
        ''' return policy values using linear softmax '''
        softmaxValues = util.Counter()
        valueSum = 0.0
        for action in self.legalActions:
            policyValue = 0.0
            for feature_name, value in self.featExtractor.getFeatures(state, action).iteritems():
                policyValue += value * self.policy_weights[feature_name]
            policyValue = math.exp(policyValue)
            softmaxValues[action] = policyValue
            valueSum += policyValue
        for action, val in softmaxValues.iteritems():
            softmaxValues[action] /= valueSum          #normalize
        return softmaxValues
    
            
    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

    def final(self, state):
        print("Training Done")
        print("Total episodes: " + str(self.episodesSoFar))
        f = open('train_weight.txt', 'w')
        for feature, weight in self.q_value_weights.iteritems():
            f.write(str(feature) + " " + str(weight) + "\n" )
        f.close()  # you can omit in most cases as the destructor will call it

        plt.plot(self.episodeRewardsList)
        plt.ylabel("Episode Reward")
        plt.show()


    #override
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
        self.episodeRewardsList.append(self.episodeRewards)
