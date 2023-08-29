import gymnasium as gym 
from gymnasium.wrappers import AtariPreprocessing, FrameStack
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
from DQN import DQNAgent


env = gym.make("ALE/Assault-v5")
env = AtariPreprocessing(env,frame_skip=1)
env = FrameStack(env, num_stack = 4)
vid = VideoRecorder(env=env,path="atari.mp4")

agent = DQNAgent(env = env,batch_size = 128,epsilon_decay = 1000, learning_rate = 1e-4,n_observations = 4,n_actions = env.action_space.n, buffer_size = 1_000_000, architecture = "DuelingDQN", device = "cpu")

# train for just 100 episodes
agent.learn(n_episodes = 100,verbose = 1)


total_rewards = []
for episode in range(10):
    state,info = env.reset()
    done = False
    total_reward = 0
    while not done:
        
        vid.capture_frame()
        
        state = agent.preprocess_state(state)
        action = agent.choose_action(state)
        env.render()
        state, reward, terminated, truncated, info = env.step(action.item())
        done = truncated or terminated
        total_reward += reward
    total_rewards.append(total_reward)
    
vid.close()
