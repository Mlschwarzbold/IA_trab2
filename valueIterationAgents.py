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


from asyncio.windows_events import INFINITE
from random import randint, randrange, shuffle
from sysconfig import get_config_var
import mdp, util
from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount=0.9, iterations=100):
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
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.new_values = util.Counter()  

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for state in mdp.getStates():
            self.values[state] = 0 
            
                
        for i in range(iterations):
            for state in mdp.getStates():
                max_value = -9999
                for action in mdp.getPossibleActions(state):
                    value = 0
                    for pair in mdp.getTransitionStatesAndProbs(state,action):
                        next_state  = pair[0]
                        prob        = pair[1]
                        reward      = mdp.getReward(state, action, pair[0])
                        next_value = self.getValue(next_state)
                        
                        #-----
                        #print("-----next_state: "   + str(next_state))
                        #print("-----prob: "         + str(prob))
                        #print("-----reward: "       + str(reward))
                        #print("-----next_value: "   + str(next_value))
                        
                        value += prob * (reward + discount * next_value)
                        
                    if(action == 'exit'):
                        #print("EXIT ACTION !!!!!!!!!!!!!")
                        value = mdp.getReward(state, action, 'TERMINAL_STATE')
                        #print("Best action: " + str(action))
                    if(value > max_value):
                        max_value = value
                        #print("Best action: " + str(action))
                #print("max_Value: " + str(max_value))
                        
                    
                self.new_values[state] = max_value
                
            for state in mdp.getStates():
                self.values[state] = self.new_values[state]
                
            
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
        #util.raiseNotDefined()
        #if(mdp.isTerminal(state)):
         #   return None
        value = 0
        for pair in self.mdp.getTransitionStatesAndProbs(state,action):
            next_state  = pair[0]
            prob        = pair[1]
            reward      = self.mdp.getReward(state, action, pair[0])
            next_value = self.getValue(next_state)
            
            value += prob * (reward + self.discount * next_value)
            
        if(action == 'exit'):
            value = self.mdp.getReward(state, action, 'TERMINAL_STATE')
        return value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        #if(mdp.isTerminal(state)):
         #   return None
        


        max_value = -9999
        best_action = "exit"
        for action in self.mdp.getPossibleActions(state):
            value = 0
            for pair in self.mdp.getTransitionStatesAndProbs(state,action):
                next_state  = pair[0]
                prob        = pair[1]
                next_value = self.getValue(next_state)
    
                value += prob * next_value
                
            if(value > max_value):
                max_value = value
                best_action = action
                   
           
        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
