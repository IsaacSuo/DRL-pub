import gymnasium as gym
import keras
import numpy as np
import time 
from typing import Any, Callable

from config.train_cfg import TrainingConfig
from policy.base import BasePolicy, DoneException

class Trainer():
    def __init__(self, 
                 model_dir: str,
                 get_run_logdir: Callable[[], Any],
                 ):
        # Set up environment
        self.env = gym.make("CartPole-v1")
        self.state_size = self.env.observation_space.shape[0] # Number of observations (CartPole)
        self.action_size = self.env.action_space.n            # Number of possible actions
        self.cb = keras.callbacks.TensorBoard(log_dir = get_run_logdir(model_dir), histogram_freq=1)
        
        
    def train(self, policy: BasePolicy, cfg: TrainingConfig):
        '''
        组合不同的策略和训练配置进行训练
        默认中每个 baseline approach 都应该使用相同的 TrainingConfig
        '''
        # record start time
        start = time.time()
        # Train Counter for weight syncing
        train_counter = 0

        # For timing training
        total_training_time = 0

        # prepare all materials before training
        policy.prepare()

        for ep in range(cfg.episode):
            state, _ = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            total_reward = 0

            # record start time
            start = time.time()

            for _ in range(500):
                action = policy.act(state, self.action_size, cfg.epsilon)
                try:
                    policy.step(
                        self.env, 
                        state, 
                        action, 
                        self.state_size, 
                        total_reward, 
                        train_counter, 
                        cfg.ba,
                        cfg.epsilon,
                        cfg.gamma,
                        cfg.epsilon_min,
                        cfg.epsilon_decay,
                        cfg.target_update_freq,
                    )
                except DoneException:
                    break
                
            # record end time and log training time
            end = time.time()
            total_training_time += end - start

            # Evaluation
            # [WriteCode]   
            eval_reward_mean, eval_reward_var = policy.evaluate(ep, cfg.episode, total_reward, cfg.epsilon)


            # Log
            policy.log(eval_reward_mean, eval_reward_var, total_reward)

            # Early Stopping Condition to avoid overfitting
            # If the evaluation reward reaches the specified threshold, stop training early.
            # The default threshold is set to 500, but you should adjust this based on observed training performance.
            if policy.early_stopping(ep, eval_reward_mean, threshold=500):
                break

        # record end time and calculate average training time per episode
        # evaluate average training time per episode
        print(f"Training time: {total_training_time/cfg.episode:.4f} seconds per episode")

        self.env.close()