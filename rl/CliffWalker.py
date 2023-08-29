from tqdm import tqdm
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
from Learners import QLearner, SARSALearner

from PolicyViz import qtable_directions_map,plot_directions_heatmap,plot_steps_and_rewards
import matplotlib.pyplot as plt

# Create the environment
learning_rate = 0.6
start_epsilon = 1.0
n_episodes = 5000
epsilon_decay = start_epsilon /(n_episodes/2)
final_epsilon = 0.1

env = gym.make("CliffWalking-v0",render_mode="rgb_array")

env = TimeLimit(RecordEpisodeStatistics(env, deque_size=n_episodes), max_episode_steps=100)
vid = VideoRecorder(env=env,path="video2.mp4")

# Applying Q-learning

qlearning_agent = QLearner(env,learning_rate = learning_rate, start_epsilon = start_epsilon, epsilon_decay = epsilon_decay
             ,final_epsilon = final_epsilon)


for episode in tqdm(range(n_episodes)): 
    observation, info = env.reset()
    done = False
    while not done:
        if(episode % 50 ==0): 
            vid.capture_frame()
        env.render()
        action = qlearning_agent.chooseAction(observation)
        next_obs,reward,terminated,truncated,info = env.step(action)
        qlearning_agent.update(observation,action,reward,next_obs,terminated)
        
        done = terminated or truncated
        observation = next_obs
    qlearning_agent.decay_epsilon()
vid.close()

q_returns = env.return_queue
q_steps = env.length_queue

# Applying SARSA 
env = gym.make("CliffWalking-v0",render_mode="rgb_array")
env = TimeLimit(RecordEpisodeStatistics(env, deque_size=n_episodes), max_episode_steps=100)

sarsa_agent = SARSALearner(env,learning_rate = learning_rate, start_epsilon = start_epsilon, epsilon_decay = epsilon_decay
             ,final_epsilon = final_epsilon)

for episode in tqdm(range(n_episodes)): 
    observation, info = env.reset()
    done = False
    while not done:
        env.render()
        action = sarsa_agent.chooseAction(observation)
        next_obs,reward,terminated,truncated,info = env.step(action)
        next_action = sarsa_agent.chooseAction(next_obs)
        sarsa_agent.update(observation,action,reward,next_obs,next_action,terminated)
        
        done = terminated or truncated
        observation = next_obs
    sarsa_agent.decay_epsilon()

sarsa_returns = env.return_queue
sarsa_steps = env.length_queue

directions = {3: "←", 2: "↓", 1: "→", 0: "↑"}
qtable_val_max,qtable_directions = qtable_directions_map(qlearning_agent.q_values,directions,(4,12))
sarsa_val_max,sarsatable_directions = qtable_directions_map(sarsa_agent.q_values,directions,(4,12))
fig, ax = plt.subplots(1,2, sharey = True, figsize = (30,10))

plot_directions_heatmap(qtable_val_max,qtable_directions,ax[0], "Q-learning" )
plot_directions_heatmap(sarsa_val_max, sarsatable_directions,ax[1],"SARSA")

fig, axs = plt.subplots(ncols = 3, figsize = (12,5))
plot_steps_and_rewards(q_returns,q_steps, qlearning_agent.training_errors,axs)
fig, axes = plt.subplots(ncols = 3, figsize = (12,5))
plot_steps_and_rewards(sarsa_returns,sarsa_steps, sarsa_agent.training_errors,axes)
plt.show()

