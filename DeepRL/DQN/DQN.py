

from common import DuelCnnPolicy, DuelMlpPolicy, DqnCnnPolicy, DqnMlpPolicy
from collections import deque, namedtuple
import torch.optim as optim
import torch.nn as nn
import math
import random
import torch
import numpy as np
import time

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

class DQNAgent:
    def __init__(self,env,
                 batch_size : int,
                 learning_rate : float,
                 n_observations : int, 
                 n_actions : int,
                 start_epsilon : float = 0.9, 
                 epsilon_decay : int = 1e6, 
                 final_epsilon : float = 0.1, 
                 discount_factor : float = 0.95, 
                 tau : float = 0.005,
                 buffer_size : int = 1e6,
                 device : str = "cuda", 
                 policy : str = "MLPPolicy", 
                 architecture : str = "DQN",
                 doubledqn : bool = True):
        """
        Deep Q-Network implemented using Pytorch
        Parameters
        ----------
        batch_size : int 
            Number of experiences to sample from the replay buffer.
        learning_rate : float
            This is the learning rate of our agent.
        n_observations : int 
            number of frame if CNNPolicy, len(observation) if MLPPolicy
        n_actions : int
            number of actions agent can take
        discount_factor : float, optional
            The discount factor this denotes how much we value our future rewards compared to our present rewards. The default is 0.95.

        start_epsilon : float
            The epsilon we start with in our epsilon greedy policy.
            
        epsilon_decay : float
            The rate with which our epsilon decays over a given episode.
        final_epsilon : float
            This is our final epsilon, the value that our epsilon can't get less than.
        tau : float
            The rate with which we update the target network
        policy : str 
            can be MLPPolicy or CNNPolicy
        architecture : 
            can be DQN or DuelingDQN
        doubledqn : bool
            whether or not to apply DoubleDQN
       
        """
        
        self.env = env
        self.gamma = discount_factor
        self.eps_start = start_epsilon
        self.eps_decay = epsilon_decay
        self.eps_end = final_epsilon
        self.tau = tau
        self.batch_size = batch_size
        self.device = device
        
        assert architecture == "DuelingDQN" or architecture == "DQN"
        assert policy == "CNNPolicy" or policy == "MLPPolicy"
        
        if architecture  == "DuelingDQN": 
            if policy == "CNNPolicy":
                self.policy = policy
                self.policy_net = DuelCnnPolicy(n_observations,n_actions).to(self.device)
                self.target_net = DuelCnnPolicy(n_observations,n_actions).to(self.device)
            elif policy == "MLPPolicy": 
                self.policy = policy
                self.policy_net = DuelMlpPolicy(n_observations,n_actions).to(self.device)
                self.target_net = DuelMlpPolicy(n_observations,n_actions).to(self.device)
        elif architecture == "DQN":
            if policy == "CNNPolicy":
                self.policy = policy
                self.policy_net = DqnCnnPolicy(n_observations,n_actions).to(self.device)
                self.target_net = DqnCnnPolicy(n_observations,n_actions).to(self.device)
            elif policy == "MLPPolicy": 
                self.policy = policy
                self.policy_net = DqnMlpPolicy(n_observations,n_actions).to(self.device)
                self.target_net = DqnMlpPolicy(n_observations,n_actions).to(self.device)
                
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.memory = deque([],buffer_size)
        self.steps = 0
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr = learning_rate, amsgrad = True)
        self.loss = []
        self.architecture = architecture
        self.doubledqn = doubledqn
    def choose_action(self,state):
        eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1*self.steps/self.eps_decay)
        self.steps += 1
        sample = random.random()
        if sample > eps:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else: 
            return torch.tensor([[self.env.action_space.sample()]],device = self.device, dtype = torch.long)
        
    def memorize(self,*args): 
        self.memory.append(Transition(*args))
        
    def recall(self):
        return random.sample(self.memory,self.batch_size)
    def preprocess_state(self, state):
        
        if self.policy == "CNNPolicy":
            state = torch.tensor(state.__array__(),dtype=torch.float,device = self.device).unsqueeze(0)
        else:
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return state
            
    def state_action_value(self,state,action):
        return self.policy_net(state).gather(1, action)
    
    def td_target(self,non_final_mask,non_final_next_states,reward): 
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            if self.doubledqn : 
                actions = self.policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1,actions).squeeze(1)
            else:
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
            
        return (next_state_values * self.gamma) + reward
    
    def optimize(self): 
        if len(self.memory) < self.batch_size:
            return 
        transitions = self.recall()
        
        batch = Transition(*zip(*transitions))

        
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.state_action_value(state_batch,action_batch)
        expected_state_action_values = self.td_target(non_final_mask,non_final_next_states,reward_batch)

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.loss.append(loss.item())
        # In-place gradient clipping
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
    def learn(self,n_episodes : int, verbose : int = 1, print_every : int = 10, save_every : int = None):  
        """
        function to train the DQN Agent

        Parameters
        ----------
        n_episodes : int
            The number of episodes to train the agent for.
        verbose : int, optional
            whether or not to print episode statistics. 1 to print episode statistics, anything different than 1 means not printing The default is 1.
        print_every : int, optional
            after how many episodes to print statistics. The default is 10.
        save_every : int, optional
            after how many episodes we save network's weights. The default is 100.

        Returns
        -------
        None.

        """
        self.rewards = []
        self.time_steps = deque(maxlen = n_episodes)
        start = time.time()
        for episode in range(n_episodes): 
            state,info = self.env.reset()
            state = self.preprocess_state(state)
            done = False 
            total_rewards = 0
            steps = 0
            while not done: 
                action = self.choose_action(state)
                steps += 1
                observation, reward, terminated, truncated, info = self.env.step(action.item())
                total_rewards += reward
                reward = torch.tensor([reward], device=self.device)

                if terminated:
                    next_state = None
                else:
                    if self.policy == "CNNPolicy" : 
                        next_state = self.preprocess_state(observation)
                    else:
                        next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                # Store the transition in memory
                self.memorize(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize()

                # Soft update of the target network's weights

                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
                self.target_net.load_state_dict(target_net_state_dict)
                
                done = terminated or truncated
            
            i = episode + 1 if episode + 1 < print_every else print_every
            self.rewards.append(total_rewards)
            self.time_steps.append(steps)
            if verbose == 1:
                if (episode + 1) % print_every == 0 :
                    print("-"*6)
                    print(f"Episode Steps : {steps}, Total Time Steps : {np.sum(self.time_steps)}")
                    print(f'Episode : {episode + 1}, Episode Reward : {total_rewards} Average Rewards over previous {i} episodes: {np.mean(self.rewards[episode-i:episode])}, Rewards Standard Deviation : {np.std(self.rewards)} \n')
                
            if save_every != None: 
                if episode % save_every == 0 :
                    torch.save(self.policy_net.state_dict(), f"policy_weights_{episode}")
                    torch.save(self.target_net.state_dict(),f"target_weights_{episode}")
             
        print('Complete')
        print(f"{n_episodes} episodes complete in {(time.time() - start)//3600} hours, {(time.time() - start)%3600//60} minutes and {(time.time() - start)%60} seconds.")
