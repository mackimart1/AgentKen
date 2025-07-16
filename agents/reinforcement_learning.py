import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Tuple, List, Optional, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict
import json
import pickle
import logging
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.monitor import load_results
import matplotlib.pyplot as plt
from datetime import datetime
import time
import threading  # Not strictly used in final version, but considered.

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RLConfig:
    """Configuration class for RL parameters"""

    algorithm: str = "PPO"  # PPO, A2C, DQN
    policy: str = "MlpPolicy"
    learning_rate: float = 3e-4
    batch_size: int = 64  # For PPO, n_steps * n_envs; for DQN, actual batch size
    buffer_size: int = 100000  # For DQN
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE parameter (PPO/A2C)
    clip_range: float = 0.2  # PPO clip range
    ent_coef: float = 0.01  # Entropy coefficient
    vf_coef: float = 0.5  # Value function coefficient (PPO/A2C)
    max_grad_norm: float = 0.5  # For PPO/A2C
    n_epochs: int = 10  # For PPO
    n_steps: int = 2048  # For PPO/A2C, steps per environment before update
    target_update_interval: int = (
        10000  # For DQN, target network update frequency (timesteps)
    )
    exploration_fraction: float = 0.1  # For DQN
    exploration_final_eps: float = 0.05  # For DQN
    learning_starts: int = 1000  # For DQN, timesteps before learning starts
    verbose: int = 1
    seed: int = 42
    device: str = "auto"  # "auto", "cpu", "cuda"
    log_interval: int = 1  # For PPO/A2C, log every N updates
    tensorboard_log: Optional[str] = "./rl_tensorboard_logs/"


@dataclass
class TaskState:
    """Enhanced state representation for tasks"""

    task_id: str = field(default_factory=lambda: f"task_{np.random.randint(100000)}")
    task_type: str = "generic"
    completion_status: float = 0.0  # 0.0 to 1.0
    error_count: int = 0
    success_count: int = 0  # Could represent sub-task successes
    retry_count: int = 0
    execution_time_steps: float = 0.0  # Number of steps spent on this task
    complexity_score: float = 0.5  # Normalized 0.0 to 1.0
    agent_confidence: float = (
        0.5  # Agent's own confidence before RL decision, 0.0 to 1.0
    )
    resource_usage: float = 0.0  # Normalized 0.0 to 1.0
    context_relevance: float = 0.5  # How relevant is current context/memory, 0.0 to 1.0
    creation_timestamp: float = field(default_factory=time.time)

    def to_vector(self) -> np.ndarray:
        """Convert state to numerical vector for the observation space"""
        age_in_hours = (time.time() - self.creation_timestamp) / 3600.0
        return np.array(
            [
                self.completion_status,
                min(self.error_count / 10.0, 1.0),  # Normalized error count
                min(self.success_count / 10.0, 1.0),  # Normalized success count
                min(self.retry_count / 5.0, 1.0),  # Normalized retry count
                min(
                    self.execution_time_steps / 100.0, 1.0
                ),  # Normalized execution time steps
                self.complexity_score,
                self.agent_confidence,
                self.resource_usage,
                self.context_relevance,
                min(age_in_hours / 24.0, 1.0),  # Normalized age (max 1 day)
            ],
            dtype=np.float32,
        )

    @staticmethod
    def get_observation_space_shape():
        # Corresponds to the number of elements in to_vector()
        return (10,)


class AdvancedTaskEnvironment(gym.Env):
    """Enhanced Environment for multi-dimensional task completion decision-making"""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        config: Optional[RLConfig] = None,
        initial_task_state: Optional[TaskState] = None,
    ):
        super(AdvancedTaskEnvironment, self).__init__()

        self.config = config or RLConfig()

        # Action space: [action_type, intensity_param1, intensity_param2]
        # action_type: 0=continue, 1=retry, 2=abort, 3=escalate/request_help, 4=optimize_strategy
        # intensity_param1/2: Parameters for the chosen action (e.g., effort, resource allocation)
        self.action_space = spaces.Box(
            low=np.array(
                [0, 0.0, 0.0]
            ),  # action_type is continuous, will be discretized
            high=np.array([4.99, 1.0, 1.0]),  # action_type up to 4.99 to map to 0-4
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,  # Assuming all features in TaskState.to_vector() are normalized
            shape=TaskState.get_observation_space_shape(),
            dtype=np.float32,
        )

        self.current_task: TaskState = (
            initial_task_state if initial_task_state else self._generate_random_task()
        )
        self.episode_step = 0
        self.max_episode_steps = 200  # Max steps per task/episode

        # Enhanced reward weights with better balance
        self.reward_weights = {
            # Task completion rewards
            "completion_progress": 15.0,  # Increased for better progress encouragement
            "task_completed_bonus": 100.0,  # Increased for stronger completion incentive
            "completion_quality": 20.0,  # New: Reward for high-quality completion
            # Efficiency rewards
            "efficiency": 5.0,  # Increased for better resource management
            "time_efficiency": 3.0,  # New: Reward for completing tasks quickly
            "resource_efficiency": 4.0,  # New: Reward for optimal resource usage
            # Error handling rewards
            "error_penalty": -10.0,  # Increased penalty for errors
            "error_recovery": 8.0,  # New: Reward for recovering from errors
            "error_prevention": 5.0,  # New: Reward for avoiding errors
            # Action-specific rewards
            "retry_cost": -3.0,  # Adjusted for better retry balance
            "abort_penalty_incomplete": -30.0,  # Increased penalty for incomplete abortion
            "abort_penalty_early": -15.0,  # Increased penalty for early abortion
            "escalate_cost": -5.0,  # Adjusted escalation cost
            "optimize_benefit": 3.0,  # Increased optimization benefit
            # Learning rewards
            "learning_progress": 2.0,  # New: Reward for improving agent confidence
            "context_utilization": 2.0,  # New: Reward for using context effectively
            # Time-based penalties
            "time_penalty": -0.2,  # Increased time penalty
            "stagnation_penalty": -1.0,  # New: Penalty for no progress
        }
        self._next_task_to_set: Optional[TaskState] = None
        self._last_completion_status = 0.0  # Track progress for stagnation detection

    def set_next_task(self, task_state: TaskState):
        """Allows external system to set the task for the next reset."""
        self._next_task_to_set = task_state

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)  # Important for reproducibility

        if self._next_task_to_set:
            self.current_task = self._next_task_to_set
            self._next_task_to_set = None
        elif options and "task_state" in options:
            self.current_task = options["task_state"]
        else:
            self.current_task = self._generate_random_task()

        self.episode_step = 0
        observation = self.current_task.to_vector()
        info = self._get_info()
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.episode_step += 1

        # Discretize action_type, ensure it's within [0, 4]
        action_type = int(np.clip(action[0], 0, 4))
        param1 = np.clip(action[1], 0.0, 1.0)
        param2 = np.clip(action[2], 0.0, 1.0)

        # Store previous state for reward calculation
        prev_completion_status = self.current_task.completion_status
        prev_error_count = self.current_task.error_count

        # Update task state based on action
        self._update_task_state(action_type, param1, param2)

        # Calculate reward
        reward = self._calculate_reward(
            action_type, param1, param2, prev_completion_status, prev_error_count
        )

        # Check if episode is done
        terminated = self._check_terminated(action_type)
        truncated = self.episode_step >= self.max_episode_steps

        observation = self.current_task.to_vector()
        info = self._get_info()
        info.update({"action_type": action_type, "param1": param1, "param2": param2})

        return observation, reward, terminated, truncated, info

    def _generate_random_task(self) -> TaskState:
        return TaskState(
            task_id=f"rnd_task_{np.random.randint(100000)}",
            task_type=np.random.choice(["analysis", "coding", "research", "planning"]),
            completion_status=np.random.uniform(0.0, 0.1),  # Start with low completion
            error_count=np.random.randint(0, 2),
            complexity_score=np.random.uniform(0.1, 1.0),
            agent_confidence=np.random.uniform(0.3, 0.9),
            resource_usage=np.random.uniform(0.0, 0.2),  # Start with low resource usage
            context_relevance=np.random.uniform(0.5, 1.0),
            creation_timestamp=time.time(),
        )

    def _update_task_state(self, action_type: int, param1: float, param2: float):
        self.current_task.execution_time_steps += 1
        # Simulate effects of actions
        if action_type == 0:  # Continue
            # Progress depends on params (e.g., effort) and current confidence/relevance
            progress_increment = (
                param1
                * 0.1
                * (
                    self.current_task.agent_confidence
                    + self.current_task.context_relevance
                )
                / 2
            )
            self.current_task.completion_status = min(
                1.0, self.current_task.completion_status + progress_increment
            )
            if np.random.rand() < 0.1 * param1:  # Chance of new error
                self.current_task.error_count += 1
            self.current_task.resource_usage = min(
                1.0, self.current_task.resource_usage + 0.01 * param1
            )

        elif action_type == 1:  # Retry
            self.current_task.retry_count += 1
            self.current_task.resource_usage = min(
                1.0, self.current_task.resource_usage + 0.02 * param1
            )
            # Retry might fix an error with some probability
            if self.current_task.error_count > 0 and np.random.rand() < 0.5 * param1:
                self.current_task.error_count -= 1
            # Small chance of progress if retrying correctly
            if np.random.rand() < 0.2 * param1:
                self.current_task.completion_status = min(
                    1.0, self.current_task.completion_status + 0.05 * param1
                )

        elif action_type == 2:  # Abort
            # State doesn't change much, episode will terminate.
            pass

        elif action_type == 3:  # Escalate / Request Help
            # Escalation might improve agent_confidence or context_relevance over time (not modeled here for simplicity)
            # but costs resources.
            self.current_task.resource_usage = min(
                1.0, self.current_task.resource_usage + 0.05 * param1
            )
            # May slightly increase completion status or reduce errors if help is effective
            if np.random.rand() < 0.3 * param1:  # Help is effective
                self.current_task.completion_status = min(
                    1.0, self.current_task.completion_status + 0.1 * param1
                )
                if self.current_task.error_count > 0 and np.random.rand() < 0.2:
                    self.current_task.error_count -= 1

        elif action_type == 4:  # Optimize Strategy
            # Optimizing might reduce future resource usage or improve efficiency (not directly modeled)
            # For now, let's say it improves agent_confidence for future steps
            self.current_task.agent_confidence = min(
                1.0, self.current_task.agent_confidence + 0.1 * param1
            )
            self.current_task.resource_usage = min(
                1.0, self.current_task.resource_usage + 0.01 * param1
            )  # Small cost to optimize

    def _calculate_reward(
        self,
        action_type: int,
        param1: float,
        param2: float,
        prev_completion_status: float,
        prev_error_count: int,
    ) -> float:
        reward = 0.0

        # 1. Task Completion Rewards
        progress = self.current_task.completion_status - prev_completion_status
        reward += progress * self.reward_weights["completion_progress"]

        # Quality-based completion bonus
        if self.current_task.completion_status >= 1.0:
            quality_score = (
                (1.0 - self.current_task.error_count / 10.0)  # Error-free completion
                * (1.0 - self.current_task.resource_usage)  # Resource efficiency
                * (1.0 - self.episode_step / self.max_episode_steps)  # Time efficiency
            )
            reward += self.reward_weights["task_completed_bonus"] * quality_score
            reward += self.reward_weights["completion_quality"] * quality_score

        # 2. Efficiency Rewards
        # Time efficiency (faster completion is better)
        if progress > 0:
            time_efficiency = progress / (self.episode_step + 1)
            reward += time_efficiency * self.reward_weights["time_efficiency"]

        # Resource efficiency (optimal resource usage)
        optimal_resource_usage = 0.5  # Target resource usage level
        resource_efficiency = 1.0 - abs(
            self.current_task.resource_usage - optimal_resource_usage
        )
        reward += resource_efficiency * self.reward_weights["resource_efficiency"]

        # 3. Error Handling Rewards
        new_errors = self.current_task.error_count - prev_error_count
        if new_errors > 0:
            reward += new_errors * self.reward_weights["error_penalty"]
        elif self.current_task.error_count < prev_error_count:
            # Reward for error recovery
            error_recovery = prev_error_count - self.current_task.error_count
            reward += error_recovery * self.reward_weights["error_recovery"]

        # Error prevention (reward for maintaining low error count)
        if self.current_task.error_count == 0:
            reward += self.reward_weights["error_prevention"]

        # 4. Action-specific Rewards
        if action_type == 1:  # Retry
            # Retry is more acceptable if there are errors
            retry_penalty = self.reward_weights["retry_cost"]
            if self.current_task.error_count > 0:
                retry_penalty *= 0.5  # Reduce penalty if retrying with errors
            reward += retry_penalty

        elif action_type == 2:  # Abort
            if self.current_task.completion_status < 1.0:
                # Stronger penalty for aborting incomplete tasks
                abort_penalty = self.reward_weights["abort_penalty_incomplete"]
                if self.current_task.completion_status < 0.1:
                    abort_penalty += self.reward_weights["abort_penalty_early"]
                reward += abort_penalty

        elif action_type == 3:  # Escalate
            # Escalation cost is reduced if it leads to progress
            escalate_cost = self.reward_weights["escalate_cost"]
            if progress > 0:
                escalate_cost *= 0.7  # Reduce cost if escalation helps
            reward += escalate_cost

        elif action_type == 4:  # Optimize
            # Optimization benefit scales with task complexity
            optimize_benefit = (
                self.reward_weights["optimize_benefit"]
                * self.current_task.complexity_score
            )
            reward += optimize_benefit

        # 5. Learning Rewards
        # Reward for improving agent confidence
        if self.current_task.agent_confidence > 0.5:
            confidence_improvement = self.current_task.agent_confidence - 0.5
            reward += confidence_improvement * self.reward_weights["learning_progress"]

        # Reward for effective context utilization
        if self.current_task.context_relevance > 0.7:
            reward += self.reward_weights["context_utilization"]

        # 6. Time-based Penalties
        # Basic time penalty
        reward += self.reward_weights["time_penalty"]

        # Stagnation penalty (no progress)
        if (
            abs(self.current_task.completion_status - self._last_completion_status)
            < 0.01
        ):
            reward += self.reward_weights["stagnation_penalty"]
        self._last_completion_status = self.current_task.completion_status

        return reward

    def _check_terminated(self, action_type: int) -> bool:
        if action_type == 2:  # Abort action leads to termination
            return True
        if self.current_task.completion_status >= 1.0:  # Task completed
            return True
        return False

    def _get_info(self) -> dict:
        return {
            "task_id": self.current_task.task_id,
            "task_type": self.current_task.task_type,
            "completion_status": self.current_task.completion_status,
            "error_count": self.current_task.error_count,
            "retry_count": self.current_task.retry_count,
            "episode_step": self.episode_step,
        }

    def render(self, mode="human"):
        if mode == "human":
            print(f"Step: {self.episode_step}")
            print(f"Task: {self.current_task.task_id} ({self.current_task.task_type})")
            print(
                f"  Completion: {self.current_task.completion_status:.2f}, "
                f"Errors: {self.current_task.error_count}, "
                f"Retries: {self.current_task.retry_count}"
            )
            print(
                f"  Resources: {self.current_task.resource_usage:.2f}, "
                f"Complexity: {self.current_task.complexity_score:.2f}"
            )
        else:
            # Handle other render modes or return None for unsupported modes
            return None


class TrainingProgressCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(TrainingProgressCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        if self.save_path is not None:
            self.save_path = self.log_dir / "best_model.zip"

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Log training stats (mean reward from Monitor wrapper)
            if self.model.ep_info_buffer:
                mean_reward = np.mean(
                    [ep_info["r"] for ep_info in self.model.ep_info_buffer]
                )
                mean_length = np.mean(
                    [ep_info["l"] for ep_info in self.model.ep_info_buffer]
                )
                self.logger.record("rollout/ep_rew_mean_custom", mean_reward)
                self.logger.record("rollout/ep_len_mean_custom", mean_length)

                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.save_path:
                        self.model.save(self.save_path)
                        if self.verbose > 0:
                            logger.info(
                                f"New best model saved to {self.save_path} (mean reward: {self.best_mean_reward:.2f})"
                            )
        return True


class EnhancedReinforcementLearningManager:
    def __init__(self, config: Optional[RLConfig] = None, n_envs: int = 4):
        self.config = config or RLConfig()
        self.n_envs = n_envs

        set_random_seed(self.config.seed)

        self.env: Optional[VecEnv] = None
        self.model: Optional[Union[PPO, A2C, DQN]] = None

        self.training_history = defaultdict(list)
        self.model_path_prefix = "rl_model_checkpoint"

        self._initialize_environment()
        self._initialize_model()

    def _env_fn(self):
        env = AdvancedTaskEnvironment(config=self.config)
        env = Monitor(env)  # Monitor wrapper for SB3 logging
        return env

    def _initialize_environment(self):
        try:
            if self.n_envs == 1:
                self.env = DummyVecEnv([self._env_fn])
            else:
                self.env = make_vec_env(
                    self._env_fn,
                    n_envs=self.n_envs,
                    seed=self.config.seed,
                    vec_env_cls=SubprocVecEnv if self.n_envs > 1 else DummyVecEnv,
                )
            logger.info(f"Environment initialized with {self.n_envs} instances.")
        except Exception as e:
            logger.error(f"Failed to initialize environment: {e}", exc_info=True)
            raise

    def _initialize_model(self):
        try:
            common_params = {
                "policy": self.config.policy,
                "env": self.env,
                "learning_rate": self.config.learning_rate,
                "gamma": self.config.gamma,
                "verbose": self.config.verbose,
                "seed": self.config.seed,
                "device": self.config.device,
                "tensorboard_log": self.config.tensorboard_log,
            }

            if self.config.algorithm == "PPO":
                self.model = PPO(
                    **common_params,
                    batch_size=self.config.batch_size,  # n_steps * n_envs for PPO effectively
                    n_steps=self.config.n_steps,
                    gae_lambda=self.config.gae_lambda,
                    clip_range=self.config.clip_range,
                    ent_coef=self.config.ent_coef,
                    vf_coef=self.config.vf_coef,
                    max_grad_norm=self.config.max_grad_norm,
                    n_epochs=self.config.n_epochs,
                )
            elif self.config.algorithm == "A2C":
                self.model = A2C(
                    **common_params,
                    n_steps=self.config.n_steps,  # SB3 A2C uses n_steps per env for update
                    gae_lambda=self.config.gae_lambda,  # A2C can use GAE
                    ent_coef=self.config.ent_coef,
                    vf_coef=self.config.vf_coef,
                    max_grad_norm=self.config.max_grad_norm,
                )
            elif self.config.algorithm == "DQN":
                self.model = DQN(
                    **common_params,
                    buffer_size=self.config.buffer_size,
                    batch_size=self.config.batch_size,  # Actual batch size for DQN
                    exploration_fraction=self.config.exploration_fraction,
                    exploration_final_eps=self.config.exploration_final_eps,
                    learning_starts=self.config.learning_starts,
                    target_update_interval=self.config.target_update_interval,
                    train_freq=4,  # Timesteps
                )
            else:
                raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")

            logger.info(f"Model ({self.config.algorithm}) initialized successfully.")

        except Exception as e:
            logger.error(f"Failed to initialize model: {e}", exc_info=True)
            raise

    def train(
        self,
        total_timesteps: int = 100000,
        eval_freq: int = 5000,
        checkpoint_freq: int = 10000,
        log_dir: str = "rl_training_logs",
    ) -> Dict[str, Any]:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        try:
            logger.info(
                f"Starting training for {total_timesteps} timesteps using {self.config.algorithm}."
            )
            start_time = time.time()

            # Callbacks
            eval_env = Monitor(
                AdvancedTaskEnvironment(config=self.config)
            )  # Separate eval env
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=str(Path(log_dir) / "best_model"),
                log_path=str(Path(log_dir) / "eval_logs"),
                eval_freq=max(eval_freq // self.n_envs, 1),
                deterministic=True,
                render=False,
                n_eval_episodes=5,
            )

            progress_callback = TrainingProgressCallback(
                check_freq=max(checkpoint_freq // self.n_envs, 1), log_dir=log_dir
            )

            callback_list = CallbackList([eval_callback, progress_callback])

            if self.model is None:
                raise RuntimeError("Model not initialized. Cannot start training.")

            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback_list,
                progress_bar=True,
                tb_log_name=self.config.algorithm,  # For TensorBoard
            )

            training_time = time.time() - start_time

            # Store training history (SB3 logs to TensorBoard primarily)
            self.training_history["total_timesteps"].append(total_timesteps)
            self.training_history["training_time_seconds"].append(training_time)

            logger.info(f"Training completed in {training_time:.2f} seconds.")

            # Save final model
            final_model_path = Path(log_dir) / f"{self.model_path_prefix}_final.zip"
            self.save_model(str(final_model_path))

            return {
                "training_time": training_time,
                "total_timesteps": total_timesteps,
                "final_model_path": str(final_model_path),
                "log_dir": log_dir,
            }

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            raise

    def predict_action(
        self, task_state: TaskState, deterministic: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self.model is None:
            logger.error("Model not initialized. Cannot predict.")
            # Return a default safe action (e.g., continue with medium params)
            return np.array([0, 0.5, 0.5], dtype=np.float32), {
                "error": "Model not initialized"
            }

        observation = task_state.to_vector().reshape(
            1, -1
        )  # SB3 expects batch dimension

        action, _states = self.model.predict(observation, deterministic=deterministic)

        # Action is a Box, so it's already a np.ndarray. For single env prediction, remove batch dim.
        action_values = action[0]

        action_type_raw, param1, param2 = (
            action_values[0],
            action_values[1],
            action_values[2],
        )
        action_type_discrete = int(np.clip(action_type_raw, 0, 4))

        metadata = {
            "action_type_discrete": action_type_discrete,
            "param1": float(param1),
            "param2": float(param2),
            "raw_action": action_values.tolist(),
            "deterministic_prediction": deterministic,
            "task_id": task_state.task_id,
            "prediction_timestamp": datetime.now().isoformat(),
        }

        return action_values, metadata  # Return the continuous action from the model

    def set_task_for_env(self, task_state: TaskState, env_idx: int = 0):
        """Sets the task for a specific environment in the VecEnv for the next reset."""
        if self.env is None:
            logger.error("Environment not initialized.")
            return
        try:
            # This method is specific to how VecEnv allows interaction.
            # For DummyVecEnv, you can access sub-environments. For SubprocVecEnv, use env_method.
            if isinstance(self.env, DummyVecEnv):
                if 0 <= env_idx < self.n_envs:
                    env = self.env.envs[env_idx]
                    if hasattr(env, 'set_next_task'):
                        # Use getattr to avoid type checker issues
                        getattr(env, 'set_next_task')(task_state)
                    else:
                        logger.warning(f"Environment {env_idx} does not have set_next_task method")
                else:
                    logger.warning(
                        f"Invalid env_idx {env_idx} for {self.n_envs} environments."
                    )
            elif isinstance(self.env, SubprocVecEnv):
                # This would call `set_next_task` on the env_idx-th sub-process environment
                self.env.env_method("set_next_task", task_state, indices=[env_idx])
            else:
                logger.warning(
                    f"set_task_for_env not implemented for VecEnv type: {type(self.env)}"
                )

        except Exception as e:
            logger.error(f"Failed to set task for env {env_idx}: {e}", exc_info=True)

    def save_model(self, path: str):
        if self.model is None:
            logger.error("No model to save.")
            return
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.model.save(path)

            metadata = {
                "config": self.config.__dict__,
                "training_history_summary": {
                    k: v[-1] if v else None for k, v in self.training_history.items()
                },
                "save_timestamp": datetime.now().isoformat(),
                "model_class": self.model.__class__.__name__,
                "n_envs": self.n_envs,
            }
            metadata_path = Path(path).with_suffix(".meta.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info(
                f"Model and metadata saved successfully to {path} and {metadata_path}"
            )
        except Exception as e:
            logger.error(f"Failed to save model: {e}", exc_info=True)
            raise

    def load_model(self, path: str, config_override: Optional[RLConfig] = None):
        try:
            model_path = Path(path)
            metadata_path = model_path.with_suffix(".meta.json")

            if config_override:
                self.config = config_override
            elif metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                self.config = RLConfig(
                    **metadata.get("config", {})
                )  # Load config from metadata
                self.n_envs = metadata.get(
                    "n_envs", self.n_envs
                )  # Load n_envs if available
                logger.info(f"Loaded config from metadata: {self.config}")

            # Re-initialize env and model structure before loading weights
            self._initialize_environment()  # Env needs to match the one used for training

            if self.config.algorithm == "PPO":
                self.model = PPO.load(
                    model_path, env=self.env, device=self.config.device
                )
            elif self.config.algorithm == "A2C":
                self.model = A2C.load(
                    model_path, env=self.env, device=self.config.device
                )
            elif self.config.algorithm == "DQN":
                self.model = DQN.load(
                    model_path, env=self.env, device=self.config.device
                )
            else:
                raise ValueError(
                    f"Unsupported algorithm for loading: {self.config.algorithm}"
                )

            logger.info(f"Model loaded successfully from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise


def plot_training_progress(log_dir: str = "rl_training_logs"):
    """Plots training progress from SB3 Monitor files if available."""
    from stable_baselines3.common.monitor import load_results
    from stable_baselines3.common.results_plotter import ts2xy

    monitor_files = list(
        Path(log_dir).glob("**/monitor.csv")
    )  # Search in eval_logs or others
    if not monitor_files:
        # Try to find monitor files in the direct log_dir as well if eval_logs is not used/present
        monitor_files = list(Path(log_dir).glob("monitor.csv"))

    if not monitor_files:
        logger.warning(
            f"No monitor.csv files found in {log_dir} or its subdirectories."
        )
        return

    # For simplicity, plot the first monitor file found. A more robust solution
    # would aggregate or allow selection.
    monitor_file_path = str(monitor_files[0].parent)

    x, y = ts2xy(
        load_results(monitor_file_path), "timesteps"
    )  # ep_rew_mean comes from Monitor logs

    if len(x) == 0:
        logger.warning(f"No data to plot from {monitor_file_path}.")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(x, y)
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Episode Reward")
    plt.title("Training Progress")
    plt.grid(True)

    plot_path = Path(log_dir) / "training_progress.png"
    plt.savefig(plot_path)
    logger.info(f"Training progress plot saved to {plot_path}")
    plt.show()


if __name__ == "__main__":
    logger.info("Starting Reinforcement Learning module test...")

    # Configuration for testing
    test_config = RLConfig(
        algorithm="PPO",
        learning_rate=1e-4,  # Smaller LR for stability with small test
        n_steps=128,  # Smaller n_steps for faster updates in test
        batch_size=64,
        n_epochs=4,
        tensorboard_log="./rl_test_tb_logs/",
        verbose=1,
    )

    # Test with 1 environment for simplicity
    rl_manager = EnhancedReinforcementLearningManager(config=test_config, n_envs=1)

    # Test training
    LOG_DIR = "rl_test_training_output"
    try:
        logger.info("--- Testing Training ---")
        # Short training run for testing
        training_results = rl_manager.train(
            total_timesteps=2000, eval_freq=500, checkpoint_freq=1000, log_dir=LOG_DIR
        )
        logger.info(f"Training results: {training_results}")

        # Test prediction
        logger.info("\n--- Testing Prediction ---")
        sample_task_state = TaskState(task_id="test_predict_task", complexity_score=0.7)
        action, metadata = rl_manager.predict_action(sample_task_state)
        logger.info(f"Predicted action: {action}, Metadata: {metadata}")

        # Test saving model
        logger.info("\n--- Testing Model Saving ---")
        save_path = Path(LOG_DIR) / "test_model_final_save"
        rl_manager.save_model(str(save_path))

        # Test loading model
        logger.info("\n--- Testing Model Loading ---")
        # Create a new manager instance to ensure it loads fresh
        loaded_rl_manager = EnhancedReinforcementLearningManager(
            config=test_config, n_envs=1
        )  # Config will be overridden by metadata if present
        loaded_rl_manager.load_model(
            str(save_path.with_suffix(".zip"))
        )  # Pass .zip path

        loaded_action, loaded_metadata = loaded_rl_manager.predict_action(
            sample_task_state
        )
        logger.info(
            f"Loaded model predicted action: {loaded_action}, Metadata: {loaded_metadata}"
        )

        assert np.allclose(
            action, loaded_action, atol=1e-5
        ), "Action from loaded model differs!"
        logger.info("Prediction from original and loaded model are consistent.")

        # Test plotting (if matplotlib is available and data exists)
        try:
            logger.info("\n--- Testing Plotting ---")
            plot_training_progress(LOG_DIR)
        except Exception as e:
            logger.warning(
                f"Plotting test failed (matplotlib might be missing or no data): {e}"
            )

    except Exception as e:
        logger.error(f"An error occurred during testing: {e}", exc_info=True)
    finally:
        # Clean up tensorboard logs if any were created in the CWD
        tb_log_path = Path(test_config.tensorboard_log or "./rl_test_tb_logs/")
        # if tb_log_path.exists() and tb_log_path.is_dir():
        #     import shutil
        #     shutil.rmtree(tb_log_path)
        #     logger.info(f"Cleaned up tensorboard log directory: {tb_log_path}")
        logger.info("Reinforcement Learning module test finished.")
