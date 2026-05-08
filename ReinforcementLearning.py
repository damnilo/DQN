import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib import animation
from tqdm import tqdm

class VideoRecorder(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.frames = []

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        self.frames = [self.env.render()]
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        self.frames.append(self.env.render())
        return observation, reward, terminated, truncated, info

    def play(self):
        fig, ax = plt.subplots(figsize=(5, 5), layout = "constrained")
        ax.axis("off")
        img = ax.imshow(self.frames[0])

        anim = animation.FuncAnimation(
            fig, lambda i: img.set_data(self.frames[i]),
            frames=len(self.frames), interval=20,
        )
        plt.close()
        return HTML(anim.to_jshtml(default_mode="once"))

def visualize_matrix(matrix: np.ndarray, name=''):
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.title(name)
    plt.yticks(np.arange(matrix.shape[0]))
    plt.xticks(np.arange(matrix.shape[1]))
    ax.imshow(matrix, cmap=plt.cm.Blues)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, np.round(matrix[i, j], 2), ha="center", va="center", color="black")


class Q_Agent():
    def __init__(self, env: gym.Env, alpha:int, gamma:int, epsilon:int, n_episodes:int, seed:int=42):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_episodes = n_episodes
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))
        np.random.seed(seed)

        self.epsilon_decay = 0.99
        self.min_epsilon = 0.01

        self._hole_states = self.find_hole_states()

    def find_hole_states(self):
        desc = None
        if hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'desc'):
            desc = self.env.unwrapped.desc
        elif hasattr(self.env, 'desc'):
            desc = self.env.desc

        if desc is None:
            return []

        desc_arr = np.asarray(desc, dtype='U1')
        hole_positions = np.argwhere(desc_arr == 'H')
        ncol = desc_arr.shape[1]
        return [int(r * ncol + c) for r, c in hole_positions]

    def learn(self):
        rewards = []
        epsilon = self.epsilon

        for episode in tqdm(range(self.n_episodes), desc="Training"):
            state, _ = self.env.reset()
            done = False

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            while not done:
                if np.random.uniform(0, 1) < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q_table[state, :])

                next_state, reward, terminated, truncated, _ = self.env.step(action)

                if terminated and next_state in self._hole_states:
                    reward -= 1.0

                self.q_table[state, action] = self.q_table[state, action] + self.alpha * (reward + self.gamma * np.max(self.q_table[next_state, :]) - self.q_table[state, action])

                state = next_state
                done = terminated or truncated
                rewards.append(reward)

        return rewards

    def predict(self, state:np.ndarray):
        return int(np.argmax(self.q_table[state, :]))
    
    def n_actions(self):
        return self.env.action_space.n
    
    def n_states(self):
        return self.env.observation_space.n

    def get_v_table(self):
        nrow = ncol = int(np.sqrt(self.env.observation_space.n))
        return np.max(self.q_table, axis=1).reshape((nrow, ncol))

    def test_agent(self, n_episodes:int):
        total_steps = 0
        total_reward = 0.0

        for episode in tqdm(range(n_episodes), desc="Training"):
            state, _ = self.env.reset()
            done = False
            ep_steps = 0
            ep_rewards = 0.0

            while not done:
                action = self.predict(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                state = next_state
                done = terminated or truncated
                ep_steps += 1
                ep_rewards += reward

            total_steps += ep_steps
            total_reward += ep_rewards

        avg_n_steps = total_steps / n_episodes
        avg_n_rewards = total_reward / n_episodes

        return avg_n_rewards, avg_n_steps

if __name__ == "__main__":

    MAP_SIZE = "4x4"

    train_env = gym.make('FrozenLake-v1', desc=None, map_name=MAP_SIZE, is_slippery=False, render_mode='rgb_array')

    train_env = VideoRecorder(train_env)

    agent = Q_Agent(
        env=train_env,
        alpha=0.5,
        gamma=0.99,
        epsilon=1.0,
        n_episodes=500,
        seed=42
    )

    train_rewards = agent.learn()

    print("\nQ-tabla nakon treninga: ")
    print(np.round(agent.q_table, 2))

    visualize_matrix(agent.get_v_table(), name='V-table')
    visualize_matrix(agent.q_table, name='Q-table')

    agent.env = gym.make(
        'FrozenLake-v1', desc=None, map_name=MAP_SIZE,
        is_slippery=False, render_mode='human'
    )

    avg_steps, avg_reward = agent.test_agent(n_episodes=10)
    print(f"\nTest rezultati ({10} epizoda):")
    print(f" Prosecan broj koraka: {avg_steps:.2f}")
    print(f" Prosecna nagrada: {avg_reward:.2f}")

    agent.env.close()
    train_env.close()
    