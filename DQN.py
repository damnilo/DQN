import gymnasium as gym
from matplotlib.colors import LinearSegmentedColormap
import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class RewardTracker(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []

    def _on_step(self) -> bool:
        self.episode_rewards.extend(
            info['episode']['r'] for info in self.locals['infos'] if 'episode' in info
        )
        return True
    
def plot_rewards(rewards, title='', window_size=100):
    plt.figure(figsize=(10, 4))
    plt.plot(rewards, alpha=0.35, label='Episode Rewards')
    plt.title(title)

    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
        plt.plot(
            np.arange(window_size - 1, len(rewards)),
            moving_avg, 
            linewidth=3,
            label=f'Moving Average ({window_size} episodes)'
        )

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(alpha=0.3)
    plt.legend(loc='upper left')
    plt.show()

def evaluate(model, env, n_episodes=100, **env_kwargs):
    env = gym.make(env, **env_kwargs)
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        rewards.append(episode_reward)
    
    env.close()
    return rewards

def task1():
    configs = [
        ("baseline", 0.6, 0.05, 1e-3, 250, 15_000),
        ("more_explore", 0.9, 0.05, 1e-3, 250, 15_000),
        ("less_explore", 0.3, 0.05, 1e-3, 250, 15_000),
        ("high_final_eps", 0.6, 0.3, 1e-3, 250, 15_000),
        ("low_lr", 0.6, 0.05, 1e-4, 250, 15_000),
        ("high_lr", 0.6, 0.05, 1e-2, 250, 15_000),
        ("fast_target", 0.6, 0.05, 1e-3, 50, 15_000),
        ("slow_target", 0.6, 0.05, 1e-3, 500, 15_000),
        ("more_steps", 0.6, 0.05, 1e-3, 250, 30_000),
    ]

    for name, expl_fracs, expl_final, lr, target_update, total_steps in configs:
        env = gym.make('FrozenLake-v1', is_slippery=False)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        cb = RewardTracker()

        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=lr,
            learning_starts=100,
            target_update_interval=target_update,
            exploration_fraction=expl_fracs,
            exploration_initial_eps=1.0,
            exploration_final_eps=expl_final,
            seed=42, verbose=0
        )

        model.learn(total_timesteps=total_steps, callback=cb)
        plot_rewards(cb.episode_rewards, title=f'{name} - Training Rewards', window_size=100)

        eval_rewards = evaluate(model, 'FrozenLake-v1', is_slippery=False)
        plot_rewards(eval_rewards, title=f'{name} - Evaluation Rewards', window_size=10)

        env.close()

def task2():
    for slippery, label in [(True, 'Slippery'), (False, 'Non-Slippery')]:
        env = gym.make('FrozenLake-v1', is_slippery=slippery)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        cb = RewardTracker()

        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=1e-3,
            learning_starts=100,
            target_update_interval=250,
            exploration_fraction=0.8,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            seed=42, verbose=0
        )

        model.learn(total_timesteps=30_000, callback=cb)
        plot_rewards(cb.episode_rewards, title=f'{label} - Training Rewards', window_size=100)

        eval_rewards = evaluate(model, 'FrozenLake-v1', is_slippery=slippery)
        plot_rewards(eval_rewards, title=f'{label} - Evaluation Rewards', window_size=10)

        env.close()

class OneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        n=env.observation_space.n
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(n,), dtype=np.float32
        )

    def observation(self, obs):
        one_hot = np.zeros(self.observation_space.shape, dtype=np.float32)
        one_hot[obs] = 1.0
        return one_hot
    
def task3():
    env = gym.make('FrozenLake-v1', is_slippery=False)
    env = OneHotWrapper(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    cb = RewardTracker()

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        learning_starts=100,
        target_update_interval=250,
        exploration_fraction=0.6,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        seed=42, verbose=0
    )

    model.learn(total_timesteps=15_000, callback=cb)
    plot_rewards(cb.episode_rewards, title='One-Hot - Training Rewards', window_size=100)

    eval_env = gym.make('FrozenLake-v1', is_slippery=False)
    eval_env = OneHotWrapper(eval_env)
    rewards = []
    for _ in range(100):
        obs, _ = eval_env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            episode_reward += reward
            done = terminated or truncated
        rewards.append(episode_reward)
    eval_env.close()
    plot_rewards(rewards, title='One-Hot - Evaluation Rewards', window_size=10)
    env.close()

class CustomMLP(gym.ObservationWrapper):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int=64):
        super().__init__(observation_space, features_dim)
        in_dim = int(np.prod(observation_space.shape))

        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, features_dim),
            nn.ReLU()
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
    
def task4():
    env = gym.make('FrozenLake-v1', is_slippery=False)
    env = OneHotWrapper(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    cb = RewardTracker()

    policy_kwargs = dict(
        features_extractor_class=CustomMLP,
        features_extractor_kwargs=dict(features_dim=64),
        not_arch = []
    )

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        learning_starts=100,
        target_update_interval=250,
        exploration_fraction=0.6,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        seed=42, verbose=0
    )

    model.learn(total_timesteps=15_000, callback=cb)
    plot_rewards(cb.episode_rewards, title='Custom MLP - Training Rewards', window_size=100)

    eval_env = gym.make('FrozenLake-v1', is_slippery=False)
    eval_env = OneHotWrapper(eval_env)
    rewards = []
    for _ in range(100):
        obs, _ = eval_env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            episode_reward += reward
            done = terminated or truncated
        rewards.append(episode_reward)

    plot_rewards(rewards, title='Custom MLP - Evaluation Rewards', window_size=10)
    env.close()

def task5():
    env = gym.make('CartPole-v1')
    env = gym.wrappers.RecordEpisodeStatistics(env)
    cb = RewardTracker()

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        learning_starts=1000,
        batch_size=64,
        target_update_interval=500,
        exploration_fraction=0.2,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.02,
        train_freq=4,
        gradient_steps=1,
        seed=42, verbose=0
    )

    model.learn(total_timesteps=50_000, callback=cb)
    plot_rewards(cb.episode_rewards, title='CartPole - Training Rewards', window_size=50)

    eval_rewards = evaluate(model, 'CartPole-v1')
    plot_rewards(eval_rewards, title='CartPole - Evaluation Rewards', window_size=5)
    env.close()

class CartPoleWrapper(gym.ObservationWrapper):
    def __init__(self, env, angle_limit=0.2095, pos_limit=2.4):
        super().__init__(env)
        self.angle_limit = angle_limit
        self.pos_limit = pos_limit

    def reward(self, reward):
        _, _, angle, _ = self.env.unwrapped.state
        pos = self.env.unwrapped.state[0]
        shaped = (
            1.0
            - (abs(angle) / self.angle_limit)
            - 0.2 * abs(pos) / self.pos_limit
        )

        return float(shaped)
    
def task6():
    for use_shaping, label in [(False, 'No Shaping'), (True, 'With Shaping')]:
        env = gym.make('CartPole-v1')
        if use_shaping:
            env = CartPoleWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        cb = RewardTracker()

        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=1e-3,
            learning_starts=1000,
            batch_size=64,
            target_update_interval=500,
            exploration_fraction=0.2,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.02,
            train_freq=4,
            gradient_steps=1,
            seed=42, verbose=0
        )

        model.learn(total_timesteps=50_000, callback=cb)
        plot_rewards(cb.episode_rewards, title=f'{label} - Training Rewards', window_size=50)

        eval_rewards = evaluate(model, 'CartPole-v1')
        plot_rewards(eval_rewards, title=f'{label} - Evaluation Rewards', window_size=5)
        env.close()

def visualize_path(model, env, title='', n_episodes=5, use_one_hot=False, **env_kwargs):
    raw_env = gym.make(env, **env_kwargs)
    desc = raw_env.unwrapped.desc
    nrow, ncol = desc.shape
    raw_env.close()

    TITLE_COLOR = {b'S': 'green', b'G': 'blue', b'H': 'red', b'F': 'black'}
    ARROW = {
        0: '→',
        1: '↓',
        2: '←',
        3: '↑'
    }

    visit_counts = np.zeros((nrow, ncol), dtype=int)
    all_paths = []

    for _ in range(n_episodes):
        episode_env = gym.make(env, **env_kwargs)
        if use_one_hot:
            episode_env = OneHotWrapper(episode_env)
        obs, _ = episode_env.reset()
        done = False
        path = []

        while not done:
            state_idx = obs if not use_one_hot else np.argmax(obs)
            visit_counts[state_idx // ncol, state_idx % ncol] += 1
            path.append(obs)

            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            obs, _, terminated, truncated, _ = episode_env.step(action)
            done = terminated or truncated

        state_idx = obs if not use_one_hot else np.argmax(obs)
        path.append(state_idx)
        visit_counts[state_idx // ncol, state_idx % ncol] += 1
        all_paths.append(path)
        episode_env.close()

    last_path = all_paths[-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title)
    ax = axes[0]
    ax.set_title('Visit Counts')
    ax.set_xlim(-0.5, ncol - 0.5)
    ax.set_ylim(-0.5, nrow - 0.5)
    ax.set_xticks(np.arange(ncol))
    ax.set_yticks(np.arange(nrow))
    ax.grid(True, linewidth=0.5, alpha=0.4)

    for r in range(nrow):
        for c in range(ncol):
            tile = bytes(desc[r, c])
            color = TITLE_COLOR.get(tile, 'black')
            ax.add_patch(mpatches.FancyBboxPatch(
                (c - 0.45, r - 0.45), 0.9, 0.9, 
                boxstyle='square,pad=0', 
                linewidth=0, facecolor=color, zorder=0))
            ax.text(c, r, str(visit_counts[r, c]), ha='center', 
                    va='center', color='white', zorder=1)

    for step_i in range(len(last_path) - 1):
        s_from = last_path[step_i]
        s_to = last_path[step_i + 1]
        r0, c0 = divmod(s_from, ncol)
        r1, c1 = divmod(s_to, ncol)

        if(r0,c0) != (r1,c1):
            ax.annotate(
                '', xy=(c1, r1), xytext=(c0, r0),
                arrowprops=dict(arrowstyle='->', color='yellow', lw=2),
                zorder = 3
            )

    sr, sc = divmod(last_path[0], ncol)
    er, ec = divmod(last_path[-1], ncol)
    ax.plot(sc, sr, 'go', markersize=12, label='Start', zorder=4)
    ax.plot(ec, er, 'bo', markersize=12, label='Goal', zorder=4)
    ax.legend(loc='lower right', fontsize = 8)

    ax2 = axes[1]
    ax2.set_title('Last Episode Path')

    heat = visit_counts.reshape(nrow, ncol).astype(float)

    cmap = LinearSegmentedColormap.from_list('visit_cmap', ['white', 'orange', 'red'])
    im = ax2.imshow(heat, cmap=cmap, vmin=0, vmax=np.max(visit_counts))
    plt.colorbar(im, ax=ax2)

    for r in range(nrow):
        for c in range(ncol):
            tile = bytes(desc[r, c])
            count = int(heat[r, c])
            ax2.text(c, r, f'{count}\n{tile.decode()}', ha='center', va='center', color='black')

    ax2.set_xticks(np.arange(ncol))
    ax2.set_yticks(np.arange(nrow))

    plt.tight_layout()
    plt.show()

def train_and_show(env, title, total_timesteps=15_000, use_one_hot=False, is_slippery=False, map_name='4x4'):
    env = gym.make(env, map_name=map_name, is_slippery=is_slippery)
    if use_one_hot:
        env = OneHotWrapper(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    cb = RewardTracker()

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        learning_starts=100,
        target_update_interval=250,
        exploration_fraction=0.6,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        seed=42, verbose=0
    )

    model.learn(total_timesteps=total_timesteps, callback=cb)
    plot_rewards(cb.episode_rewards, title=f'{title} - Training Rewards', window_size=100)

    visualize_path(model, env.spec.id, title=f'{title} - Learned Path', use_one_hot=use_one_hot, map_name=map_name, is_slippery=is_slippery)
    env.close()

    return model

if __name__ == "__main__":
    # train_and_show(
    #     "FrozenLake-v1",
    #     title="FrozenLake 4x4 – non-slippery",
    #     map_name="4x4", is_slippery=False
    # )

    # # 4x4 slippery
    # train_and_show(
    #     "FrozenLake-v1",
    #     title="FrozenLake 4x4 – slippery",
    #     map_name="4x4", is_slippery=True,
    #     total_timesteps=30_000
    # )

    # 4x4 one-hot
    train_and_show(
        "FrozenLake-v1",
        title="FrozenLake 4x4 – one-hot wrapper",
        map_name="4x4", is_slippery=False,
        use_one_hot=True
    )

    # 8x8
    train_and_show(
        "FrozenLake-v1",
        title="FrozenLake 8x8 – non-slippery",
        map_name="8x8", is_slippery=False,
        total_timesteps=50_000
    )
