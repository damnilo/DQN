import gymnasium as gym
from matplotlib.colors import LinearSegmentedColormap
import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

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

class CustomMLP(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        in_dim = int(np.prod(observation_space.shape))
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),    nn.ReLU(),
            nn.Linear(64, features_dim), nn.ReLU()
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
        net_arch = []
    )

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        learning_starts=100,
        target_update_interval=250,
        policy_kwargs=policy_kwargs,
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

    TILE_COLOR = {b'S': 'limegreen', b'G': 'dodgerblue', b'H': 'tomato', b'F': '#e8e8e8'}
    ARROW = {0: '→', 1: '↓', 2: '←', 3: '↑'}

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
            state_idx = int(obs) if not use_one_hot else int(np.argmax(obs))
            visit_counts[state_idx // ncol, state_idx % ncol] += 1
            path.append(state_idx)
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = episode_env.step(int(action))
            done = terminated or truncated

        state_idx = int(obs) if not use_one_hot else int(np.argmax(obs))
        path.append(state_idx)
        visit_counts[state_idx // ncol, state_idx % ncol] += 1
        all_paths.append(path)
        episode_env.close()

    last_path = all_paths[-1]
    last_path_set = set(last_path)

    # --- Naučena politika za svaku ćeliju ---
    policy_arrows = np.full((nrow, ncol), '', dtype=object)
    n_states = nrow * ncol
    for s in range(n_states):
        r, c = divmod(s, ncol)
        tile = bytes(desc[r, c])
        if tile in (b'H', b'G'):
            continue
        if use_one_hot:
            obs_input = np.zeros(n_states, dtype=np.float32)
            obs_input[s] = 1.0
        else:
            obs_input = np.array(s)
        action, _ = model.predict(obs_input, deterministic=True)
        policy_arrows[r, c] = ARROW[int(action)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=13)

    # --- Lijevi plot: putanja + politika ---
    ax = axes[0]
    ax.set_title('Putanja posljednje epizode + politika')
    ax.set_xlim(-0.5, ncol - 0.5)
    ax.set_ylim(nrow - 0.5, -0.5)   # y-osa invertovana da (0,0) bude gore-lijevo
    ax.set_xticks(np.arange(ncol))
    ax.set_yticks(np.arange(nrow))
    ax.tick_params(length=0)
    ax.grid(False)

    for r in range(nrow):
        for c in range(ncol):
            tile = bytes(desc[r, c])
            color = TILE_COLOR.get(tile, '#e8e8e8')
            in_path = (r * ncol + c) in last_path_set

            rect = mpatches.FancyBboxPatch(
                (c - 0.48, r - 0.48), 0.96, 0.96,
                boxstyle='round,pad=0.04',
                linewidth=2.5 if in_path else 0.5,
                edgecolor='royalblue' if in_path else '#cccccc',
                facecolor=color, zorder=1
            )
            ax.add_patch(rect)

            # tile naziv
            ax.text(c, r - 0.28, tile.decode(), ha='center', va='center',
                    fontsize=9, color='#555', zorder=2)

            # strelica politike
            arrow = policy_arrows[r, c]
            if arrow:
                ax.text(c, r + 0.15, arrow, ha='center', va='center',
                        fontsize=18, color='#333', zorder=2)

            # visit count
            ax.text(c + 0.35, r - 0.35, str(visit_counts[r, c]),
                    ha='right', va='top', fontsize=7, color='#666', zorder=2)

    # Strelice puta
    for i in range(len(last_path) - 1):
        s0, s1 = last_path[i], last_path[i + 1]
        r0, c0 = divmod(s0, ncol)
        r1, c1 = divmod(s1, ncol)
        if (r0, c0) != (r1, c1):
            ax.annotate('', xy=(c1, r1), xytext=(c0, r0),
                        arrowprops=dict(arrowstyle='->', color='royalblue', lw=2),
                        zorder=3)

    sr, sc = divmod(last_path[0], ncol)
    er, ec = divmod(last_path[-1], ncol)
    ax.plot(sc, sr, 'go', markersize=10, label='Start', zorder=4)
    ax.plot(ec, er, 'b*', markersize=14, label='Cilj', zorder=4)
    ax.legend(loc='lower right', fontsize=8)

    # --- Desni plot: heatmap ---
    ax2 = axes[1]
    ax2.set_title('Heatmap posjeta')
    cmap = LinearSegmentedColormap.from_list('visit', ['#f7f7f7', 'orange', 'crimson'])
    im = ax2.imshow(visit_counts, cmap=cmap, vmin=0, vmax=max(visit_counts.max(), 1))
    plt.colorbar(im, ax=ax2)

    for r in range(nrow):
        for c in range(ncol):
            tile = bytes(desc[r, c])
            ax2.text(c, r, f"{visit_counts[r,c]}\n{tile.decode()}",
                     ha='center', va='center', fontsize=8, color='black')

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
    train_and_show(
        "FrozenLake-v1",
        title="FrozenLake 4x4 – non-slippery",
        map_name="4x4", is_slippery=False
    )

    # # 4x4 slippery
    # train_and_show(
    #     "FrozenLake-v1",
    #     title="FrozenLake 4x4 – slippery",
    #     map_name="4x4", is_slippery=True,
    #     total_timesteps=30_000
    # )

    # # 4x4 one-hot
    # train_and_show(
    #     "FrozenLake-v1",
    #     title="FrozenLake 4x4 – one-hot wrapper",
    #     map_name="4x4", is_slippery=False,
    #     use_one_hot=True
    # )

    # # 8x8
    # train_and_show(
    #     "FrozenLake-v1",
    #     title="FrozenLake 8x8 – non-slippery",
    #     map_name="8x8", is_slippery=False,
    #     total_timesteps=50_000
    # )
