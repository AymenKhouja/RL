import numpy as np 
from collections import defaultdict

class QLearner :
    def __init__(self, env,learning_rate : float, start_epsilon : float, epsilon_decay : float, 
                final_epsilon : float, gamma : float = 0.95 ):
        """
        Constructor for our agent, it gives our agent all the required hyperparameters.

        Parameters
        ----------
        env : TYPE 
            This is the environment our agent is going to act in, must have discrete action and observations spaces.
        learning_rate : float
            This is the learning rate of our agent.
        start_epsilon : float
            The epsilon we start with in our epsilon greedy policy.
        epsilon_decay : float
            The rate with which our epsilon decays over a given episode.
        final_epsilon : float
            This is our final epsilon, the value that our epsilon can't get less than.
        gamma : float, optional
            The discount factor this denotes how much we value our future rewards compared to our present rewards. The default is 0.95.

        Returns
        -------
        None.

        """
        self.environment = env
        self.lr = learning_rate
        self.epsilon = start_epsilon
        self.epsilon_decay = epsilon_decay 
        self.final_epsilon = final_epsilon
        self.discount_factor = gamma
        self.q_values = defaultdict(lambda : np.zeros(env.action_space.n))
        self.training_errors = []
        

    
    def chooseAction(self, state : tuple) -> int:
        """
        Our epsilon greedy policy

        Parameters
        ----------
        state : tuple
            This is the current observation of the environment.

        Returns
        -------
        int
            Action that our agent is going to take.

        """
        if(np.random.uniform() < self.epsilon): 
            return self.environment.action_space.sample()
        else: 
            return int(np.argmax(self.q_values[state]))
    
    def update(self,state : tuple,action : int,reward : int ,next_state :tuple, terminated: bool):
        """
        Updates the q_table of the agent

        Parameters
        ----------
        state : tuple
            Current State.
        action : int
            Action to take.
        reward : int
            Reward after given action.
        next_state : tuple
            The state after doing the action.
        terminated : bool
            Whether or not the episode is terminated after the action.

        Returns
        -------
        None.

        """
        future_q_value = (not terminated)*max(self.q_values[next_state])
        td = reward + self.discount_factor * future_q_value - self.q_values[state][action]
        self.q_values[state][action] = self.q_values[state][action] + self.lr * td
        self.training_errors.append(td)
    
    def decay_epsilon(self):
        """
        Updates the value of our epsilon, it's choosen as the max of the final_epsilon and the current epsilon - the decay rate

        Returns
        -------
        None.

        """
        self.epsilon = max(self.final_epsilon,self.epsilon - self.epsilon_decay)


class SARSALearner :
    def __init__(self, env,learning_rate : float, start_epsilon : float, epsilon_decay : float, 
                final_epsilon : float, gamma : float = 0.95 ):
        """
        Constructor for our agent, it gives our agent all the required hyperparameters.

        Parameters
        ----------
        env : TYPE 
            This is the environment our agent is going to act in, must have discrete action and observations spaces.
        learning_rate : float
            This is the learning rate of our agent.
        start_epsilon : float
            The epsilon we start with in our epsilon greedy policy.
        epsilon_decay : float
            The rate with which our epsilon decays over a given episode.
        final_epsilon : float
            This is our final epsilon, the value that our epsilon can't get less than.
        gamma : float, optional
            The discount factor this denotes how much we value our future rewards compared to our present rewards. The default is 0.95.

        Returns
        -------
        None.

        """
        self.environment = env
        self.lr = learning_rate
        self.epsilon = start_epsilon
        self.epsilon_decay = epsilon_decay 
        self.final_epsilon = final_epsilon
        self.discount_factor = gamma
        self.q_values = defaultdict(lambda : np.zeros(env.action_space.n))
        self.training_errors = []
        

    
    def chooseAction(self, state : tuple[int,dict]) -> int:
        """
        Our epsilon greedy policy

        Parameters
        ----------
        state : tuple[int,dict]
            This is the current observation of the environment.

        Returns
        -------
        int
            Action that our agent is going to take.

        """
        if(np.random.uniform() < self.epsilon): 
            return self.environment.action_space.sample()
        else: 
            return int(np.argmax(self.q_values[state]))
    
    def update(self,state : tuple[int,dict],action : int,reward : int ,next_state :tuple[int,dict], next_action : int, terminated: bool):
        """
        Updates the q_table of the agent

        Parameters
        ----------
        state : tuple[int,dict]
            Current State.
        action : int
            Action to take.
        reward : int
            Reward after given action.
        next_state : tuple[int,dict]
            The state after doing the action.
        next_action : int
            action to take in the next_state
        terminated : bool
            Whether or not the episode is terminated after the action.

        Returns
        -------
        None.

        """
        future_q_value = (not terminated)*self.q_values[next_state][next_action]
        td = reward + self.discount_factor * future_q_value - self.q_values[state][action]
        self.q_values[state][action] = self.q_values[state][action] + self.lr * td
        self.training_errors.append(td)
    
    def decay_epsilon(self):
        """
        Updates the value of our epsilon, it's choosen as the max of the final_epsilon and the current epsilon - the decay rate

        Returns
        -------
        None.

        """
        self.epsilon = max(self.final_epsilon,self.epsilon - self.epsilon_decay)
