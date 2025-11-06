import gymnasium as gym
from gymnasium import Env
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from agent.core import CoreAgent
from policy.base import BasePolicy
from config.train import TrainingConfig
from keras.callbacks import TensorBoard


class QNetworkAgent(CoreAgent):
    def __init__(self, env: Env, policy: BasePolicy, cfg: TrainingConfig, cb: TensorBoard):
        super().__init__(env, policy, cfg, cb)
        # 存储训练日志
        self.train_reward_lst = []
        self.eval_reward_mean_lst = []
        self.eval_reward_var_lst = []
        self.total_training_time = 0.0

    def _evaluate(self, eval_episodes=10):
        rewards = []
        for _ in range(eval_episodes):
            state, _ = self.env.reset()
            state = np.reshape(state, [1, self.env.observation_space.shape[0]])
            total_reward = 0
            while True:
                action = self.policy.select_action(state, epsilon=0.0)  # greedy
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                if done:
                    break
                state = np.reshape(next_state, [1, self.env.observation_space.shape[0]])
            rewards.append(total_reward)
        return np.mean(rewards), np.var(rewards)

    def learn(self):
        cfg = self.cfg
        epsilon = cfg.epsilon
        gamma = cfg.gamma
        epsilon_min = cfg.epsilon_min
        epsilon_decay = cfg.epsilon_decay

        for ep in range(cfg.episode):
            state, _ = self.env.reset()
            state = np.reshape(state, [1, self.env.observation_space.shape[0]])
            total_reward = 0
            start = time.time()

            for step in range(500):
                action = self.policy.select_action(state, epsilon)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.env.observation_space.shape[0]])
                done = terminated or truncated

                # Q-learning target
                if done:
                    target = reward
                else:
                    next_q = self.policy.model(next_state, training=False).numpy()
                    target = reward + gamma * np.max(next_q[0])

                # Get current Q
                q_orig = self.policy.model(state, training=False).numpy()
                target_q = q_orig.copy()
                target_q[0, action] = target
                target_q = tf.constant(target_q, dtype=tf.float32)

                # Single-sample gradient update
                with tf.GradientTape() as tape:
                    q_pred = self.policy.model(state, training=True)
                    loss = tf.reduce_mean(tf.square(target_q - q_pred))

                grads = tape.gradient(loss, self.policy.model.trainable_variables)
                self.policy.optimizer.apply_gradients(zip(grads, self.policy.model.trainable_variables))

                # Decay epsilon
                if epsilon > epsilon_min:
                    epsilon *= epsilon_decay

                state = next_state
                total_reward += reward
                if done:
                    break

            end = time.time()
            self.total_training_time += (end - start)

            # Evaluation
            eval_mean, eval_var = self._evaluate()
            self.train_reward_lst.append(total_reward)
            self.eval_reward_mean_lst.append(eval_mean)
            self.eval_reward_var_lst.append(eval_var)

            print(f"Episode {ep + 1}/{cfg.episode} | "
                  f"Train Rwd: {total_reward} | "
                  f"Epsilon: {epsilon:.3f} | "
                  f"Eval Mean: {eval_mean:.2f}")

            # Early stopping (optional)
            # if eval_mean >= 500:
            #     print("Early stopping!")
            #     break

        print(f"Avg training time per episode: {self.total_training_time / cfg.episode:.4f}s")
