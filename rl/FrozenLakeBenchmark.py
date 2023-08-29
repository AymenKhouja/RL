from tqdm import tqdm
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
from Learners import QLearner
import matplotlib.pyplot as plt
from PolicyViz import qtable_directions_map,plot_directions_heatmap


learning_rate = 0.8
start_epsilon = 0.5
n_episodes = 2000
epsilon_decay = 0.00001
final_epsilon = 0.1

env = gym.make("FrozenLake-v1",render_mode="rgb_array",is_slippery=False)
env = RecordEpisodeStatistics(env, deque_size=n_episodes)
vid = VideoRecorder(env=env,path="video4.mp4")


agent = QLearner(env,learning_rate = learning_rate, start_epsilon = start_epsilon, epsilon_decay = epsilon_decay
             , final_epsilon = final_epsilon)

for episode in tqdm(range(n_episodes)): 
    observation, info = env.reset()
    done = False
    while not done:
        if episode % 50 == 0:
            vid.capture_frame()
        env.render()
        action = agent.chooseAction(observation)
        next_obs,reward,terminated,truncated,info = env.step(action)
        agent.update(observation,action,reward,next_obs,terminated)
        
        done = terminated or truncated
        observation = next_obs
    agent.decay_epsilon()
vid.close()


directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
qtable_val_max,qtable_directions = qtable_directions_map(agent.q_values,directions,(4,4))

plot_directions_heatmap(qtable_val_max,qtable_directions)
plt.show()
