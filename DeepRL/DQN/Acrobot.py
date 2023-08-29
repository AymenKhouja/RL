
# Implementing DQN on the Acrobot environment included in gymnasium

from DQN import DQNAgent
import gymnasium as gym
from rl.PolicyViz import plot_episodes_info


env = gym.make("Acrobot-v1")
obs, info = env.reset()

agent = DQNAgent(env = env,batch_size = 128,epsilon_decay = 1000, learning_rate = 1e-4,n_observations = len(obs),n_actions = env.action_space.n, memory_size = 100_000, architecture = "DuelingDQN", device = "cpu")

# training
agent.learn(10,5,1000)

plot_episodes_info(agent.rewards)
plt.show()

# evaluating and visualizing
env = gym.make("Acrobot-v1", render_mode = "human")
state, info = env.reset()
done = False
for episode in range(5):
    while not done:
        state = agent.preprocess_state(state)
        action = agent.choose_action(state)
        env.render()
        state, reward, terminated, truncated, info = env.step(action.item())
        done = truncated or terminated
