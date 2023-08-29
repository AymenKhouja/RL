from tqdm import tqdm
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
from QLearningAgent import QLearner
import matplotlib.pyplot as plt
from PolicyViz import qtable_directions_map,plot_directions_heatmap




learning_rate = 0.1
start_epsilon = 0.9
n_episodes = 2000
epsilon_decay = start_epsilon /(n_episodes/2)
final_epsilon = 0.1

env = gym.make("Taxi-v3",render_mode="rgb_array")
env = TimeLimit(RecordEpisodeStatistics(env, deque_size=n_episodes), max_episode_steps=200)
vid = VideoRecorder(env=env,path="video3.mp4")


agent = QLearner(env,learning_rate = learning_rate, start_epsilon = start_epsilon, epsilon_decay = epsilon_decay
             , final_epsilon = final_epsilon)

observation, info = env.reset()
done = False


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

