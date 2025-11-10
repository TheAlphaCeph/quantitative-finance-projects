"""
Ensemble RL Agent
=================

Combines multiple RL algorithms (PPO, SAC, TD3) for robust trading decisions.
Diversity reduces overfitting and improves generalization.

Author: Abhay Kanwar
Date: 2025-11-03
"""

import numpy as np
from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import warnings
warnings.filterwarnings('ignore')


class EnsembleAgent:
    """
    Ensemble of multiple RL algorithms with performance-based weighting.

    Combines predictions from PPO, SAC, and A2C using weighted averaging
    where weights are proportional to each model's validation Sharpe ratio.
    """

    def __init__(
        self,
        env,
        algorithms=['PPO', 'SAC', 'A2C'],
        timesteps=50000,
        verbose=0
    ):
        """
        Initialize and train ensemble of RL algorithms.

        Parameters
        ----------
        env : gym.Env
            Training environment
        algorithms : list of str
            List of algorithms to include in ensemble
            Options: 'PPO', 'SAC', 'A2C', 'TD3'
        timesteps : int
            Training timesteps per algorithm
        verbose : int
            Verbosity level
        """
        self.algorithms = algorithms
        self.models = {}
        self.weights = {}
        self.sharpes = {}
        self.verbose = verbose

        print(f"\n{'='*60}")
        print(f"ENSEMBLE AGENT - Training {len(algorithms)} algorithms")
        print(f"{'='*60}\n")

        # Train each algorithm
        for algo_name in algorithms:
            print(f"Training {algo_name}...")

            if algo_name == 'PPO':
                model = PPO(
                    "MlpPolicy",
                    env,
                    learning_rate=3e-4,
                    n_steps=2048,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    policy_kwargs=dict(net_arch=[256, 256]),
                    verbose=verbose
                )

            elif algo_name == 'SAC':
                model = SAC(
                    "MlpPolicy",
                    env,
                    learning_rate=3e-4,
                    buffer_size=100000,
                    learning_starts=1000,
                    batch_size=256,
                    tau=0.005,
                    gamma=0.99,
                    ent_coef='auto',  # Automatic entropy tuning
                    verbose=verbose
                )

            elif algo_name == 'A2C':
                model = A2C(
                    "MlpPolicy",
                    env,
                    learning_rate=3e-4,
                    n_steps=5,
                    gamma=0.99,
                    policy_kwargs=dict(net_arch=[256, 256]),
                    verbose=verbose
                )

            else:
                raise ValueError(f"Unknown algorithm: {algo_name}")

            # Train
            model.learn(total_timesteps=timesteps)
            self.models[algo_name] = model

            print(f"  ✓ {algo_name} training complete\n")

        # Evaluate and set weights
        self._calculate_weights(env)

    def _calculate_weights(self, env):
        """
        Calculate performance-based weights for each model.

        Uses a short evaluation run to estimate each model's Sharpe ratio,
        then weights proportionally to performance.
        """
        print("Evaluating models to calculate ensemble weights...")

        for algo_name, model in self.models.items():
            sharpe = self._evaluate_sharpe(model, env)
            self.sharpes[algo_name] = sharpe

            # Weight = max(0, sharpe)  (only positive performers get weight)
            self.weights[algo_name] = max(0.0, sharpe)

        # Normalize weights to sum to 1
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            for algo_name in self.weights:
                self.weights[algo_name] /= total_weight
        else:
            # If all negative, use equal weights
            uniform_weight = 1.0 / len(self.weights)
            for algo_name in self.weights:
                self.weights[algo_name] = uniform_weight

        # Print results
        print(f"\n{'='*60}")
        print("ENSEMBLE WEIGHTS")
        print(f"{'='*60}")
        for algo_name in self.algorithms:
            print(f"  {algo_name:8s}: Weight={self.weights[algo_name]:.3f}, "
                  f"Sharpe={self.sharpes[algo_name]:+.2f}")
        print(f"{'='*60}\n")

    def _evaluate_sharpe(self, model, env, n_steps=500):
        """
        Estimate Sharpe ratio for a model.

        Parameters
        ----------
        model : stable_baselines3 model
            Model to evaluate
        env : gym.Env
            Environment to evaluate on
        n_steps : int
            Number of steps for evaluation

        Returns
        -------
        sharpe : float
            Estimated Sharpe ratio
        """
        obs, _ = env.reset()
        rewards = []

        for _ in range(n_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            rewards.append(reward)

            if done or truncated:
                obs, _ = env.reset()

        rewards = np.array(rewards)

        # Sharpe ratio = mean / std * sqrt(periods_per_year)
        # Assuming hourly steps, ~8760 periods per year
        if len(rewards) > 1 and rewards.std() > 0:
            sharpe = (rewards.mean() / rewards.std()) * np.sqrt(8760)
        else:
            sharpe = 0.0

        return sharpe

    def predict(self, obs, deterministic=True):
        """
        Make ensemble prediction.

        Combines predictions from all models using weighted averaging.

        Parameters
        ----------
        obs : np.ndarray
            Current observation
        deterministic : bool
            Whether to use deterministic predictions

        Returns
        -------
        action : np.ndarray
            Ensemble action
        """
        ensemble_action = None

        for algo_name, model in self.models.items():
            action, _ = model.predict(obs, deterministic=deterministic)
            weight = self.weights[algo_name]

            if ensemble_action is None:
                ensemble_action = weight * action
            else:
                ensemble_action += weight * action

        return ensemble_action, None

    def save(self, path):
        """Save all models in ensemble."""
        import os
        os.makedirs(path, exist_ok=True)

        for algo_name, model in self.models.items():
            model_path = os.path.join(path, f"{algo_name}.zip")
            model.save(model_path)

        # Save weights
        import pickle
        weights_path = os.path.join(path, "ensemble_weights.pkl")
        with open(weights_path, 'wb') as f:
            pickle.dump({
                'weights': self.weights,
                'sharpes': self.sharpes,
                'algorithms': self.algorithms
            }, f)

        print(f"✓ Ensemble saved to {path}")

    @classmethod
    def load(cls, path, env):
        """Load ensemble from disk."""
        import os
        import pickle

        # Load weights
        weights_path = os.path.join(path, "ensemble_weights.pkl")
        with open(weights_path, 'rb') as f:
            data = pickle.load(f)

        # Create instance
        instance = cls.__new__(cls)
        instance.algorithms = data['algorithms']
        instance.weights = data['weights']
        instance.sharpes = data['sharpes']
        instance.models = {}

        # Load models
        for algo_name in instance.algorithms:
            model_path = os.path.join(path, f"{algo_name}.zip")

            if algo_name == 'PPO':
                model = PPO.load(model_path, env=env)
            elif algo_name == 'SAC':
                model = SAC.load(model_path, env=env)
            elif algo_name == 'A2C':
                model = A2C.load(model_path, env=env)

            instance.models[algo_name] = model

        print(f"✓ Ensemble loaded from {path}")
        return instance


def compare_single_vs_ensemble(env_train, env_test, timesteps=50000):
    """
    Compare single PPO vs ensemble performance.

    Parameters
    ----------
    env_train : gym.Env
        Training environment
    env_test : gym.Env
        Test environment
    timesteps : int
        Training timesteps

    Returns
    -------
    results : dict
        Comparison results
    """
    print("\n" + "="*60)
    print("SINGLE AGENT VS ENSEMBLE COMPARISON")
    print("="*60 + "\n")

    # 1. Train single PPO
    print("Training single PPO agent...")
    single_ppo = PPO(
        "MlpPolicy",
        env_train,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=0
    )
    single_ppo.learn(total_timesteps=timesteps)
    print("✓ Single PPO complete\n")

    # 2. Train ensemble
    ensemble = EnsembleAgent(
        env_train,
        algorithms=['PPO', 'SAC', 'A2C'],
        timesteps=timesteps // 3,  # Same total timesteps
        verbose=0
    )

    # 3. Evaluate both on test set
    def evaluate(agent, env, n_episodes=3):
        """Evaluate agent on environment."""
        all_rewards = []

        for ep in range(n_episodes):
            obs, _ = env.reset()
            episode_rewards = []
            done = False
            step = 0
            max_steps = 800

            while not done and step < max_steps:
                if isinstance(agent, EnsembleAgent):
                    action, _ = agent.predict(obs, deterministic=True)
                else:
                    action, _ = agent.predict(obs, deterministic=True)

                obs, reward, done_flag, truncated, _ = env.step(action)
                episode_rewards.append(reward)
                done = done_flag or truncated
                step += 1

            all_rewards.extend(episode_rewards)

        rewards = np.array(all_rewards)
        sharpe = (rewards.mean() / rewards.std()) * np.sqrt(8760) if rewards.std() > 0 else 0
        return sharpe, rewards

    print("\nEvaluating on test set...")
    single_sharpe, single_rewards = evaluate(single_ppo, env_test)
    ensemble_sharpe, ensemble_rewards = evaluate(ensemble, env_test)

    print(f"\nRESULTS:")
    print(f"  Single PPO:  Sharpe = {single_sharpe:+.2f}")
    print(f"  Ensemble:    Sharpe = {ensemble_sharpe:+.2f}")
    print(f"  Improvement: {((ensemble_sharpe / single_sharpe - 1) * 100):+.1f}%")

    return {
        'single_sharpe': single_sharpe,
        'ensemble_sharpe': ensemble_sharpe,
        'improvement': ensemble_sharpe - single_sharpe
    }


if __name__ == "__main__":
    # Test with dummy environment
    print("Ensemble Agent Module - Ready for import")
    print("Use: from cmds.ensemble_agent import EnsembleAgent")
