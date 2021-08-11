# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        for i in range(self.iterations):
            valueForState = util.Counter()

            # Bellman update for each state
            for s in self.mdp.getStates():
                valueForAction = util.Counter()
                for a in self.mdp.getPossibleActions(s):
                    valueForAction[a] = self.computeQValueFromValues(s, a)
                valueForState[s] = valueForAction[valueForAction.argMax()]
            for s in self.mdp.getStates():
                self.values[s] = valueForState[s]


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        qVal = 0
        probability = self.mdp.getTransitionStatesAndProbs(state, action)
        for nextState, prob in probability:
            qVal += prob * (self.mdp.getReward(state, action, nextState) +
                                       self.discount * self.getValue(nextState))
        return qVal
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
          V(s) = max_{a in actions} Q(s,a)
        """

        "*** YOUR CODE HERE ***"
        # Get all the possible actions from the state
        actions = self.mdp.getPossibleActions(state)

        # if no actions then return
        if not actions:
            return None

        best_action, best_reward = '', -1*float('inf')

        # Compute the best action
        for a in actions:
            probability = self.mdp.getTransitionStatesAndProbs(state, a) #Returns list of (nextState, prob) pairs
            curr_reward = 0

            # Action that gives the highest reward is the best action
            # Compute the reward for each nextState, prob pair
            for nextState, prob in probability:
                curr_reward += prob * (self.mdp.getReward(state, a, nextState) +
                                       self.discount * self.getValue(nextState))

            # return action with the highest reward
            if curr_reward > best_reward:
                best_action = a
                best_reward = curr_reward

        return best_action
        #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
