import gymnasium as gym
from gymnasium import Env
import numpy as np
import tensorflow as tf
import keras
import time 
import matplotlib.pyplot as plt
import json
        
from policy.base import BasePolicy
from config.train import TrainingConfig
from keras.callbacks import TensorBoard

class CoreAgent():
    def __init__(self, env: Env, policy: BasePolicy, cfg: TrainingConfig, cb: TensorBoard):
        self.policy = policy
        self.env = env
        self.cfg = cfg
        self.cb = cb
        self.train_reward_lst = []
        self.eval_reward_mean_lst = []
        self.eval_reward_var_lst = []
        
    def act(self, observation, deterministic: bool, verbose=0):
        '''
        选择并执行动作
        deterministic 为 False 时采取 epsilon-greedy 探索策略
        deterministic 为 True 时采取模型推理动作
        '''
        state, action_size, epsilon = observation
        if np.random.rand() <= epsilon and not deterministic:
            action = np.random.choice(action_size)
        else:
            q_values = self.policy.get_action(state, verbose=0)
            action = np.argmax(q_values)
        return action
    
    def collect(self, total_reward, epsilon, eval_reward_mean, eval_reward_var, ep, episode):
        '''收集一个完整回合的数据'''
        self.train_reward_lst.append(total_reward)
        self.eval_reward_mean_lst.append(eval_reward_mean)
        self.eval_reward_var_lst.append(eval_reward_var)
        print(f"Episode {ep + 1}/{episode} | Ep. Total Reward: {total_reward}| Epsilon : {epsilon:.3f}| Eval Rwd Mean: {eval_reward_mean:.2f}| Eval Rwd Var: {eval_reward_var:.2f}")

        # TensorBoard实时日志记录
        if self.cb:
            with tf.summary.create_file_writer(self.cb.log_dir).as_default():
                tf.summary.scalar('Training/Episode_Total_Reward', total_reward, step=ep)
                tf.summary.scalar('Training/Epsilon', epsilon, step=ep)
                tf.summary.scalar('Evaluation/Reward_Mean', eval_reward_mean, step=ep)
                tf.summary.scalar('Evaluation/Reward_Variance', eval_reward_var, step=ep)
    
    def evaluate(self, max_timesteps):
        '''通过新建重置环境 评估当前策略的表现'''
        eval_env = gym.make("CartPole-v1")
        state_size = eval_env.observation_space.shape[0]
        action_size = eval_env.action_space.n
        eval_reward = []

        for i in range (5):
            round_reward = 0
            state, _ = eval_env.reset()
            state = np.reshape(state, [1, state_size])

            for i in range(max_timesteps):
                # action = np.argmax(model.predict(state, verbose=0)[0])
                observation = (state, action_size, 1)
                action = self.act(observation, deterministic=True, verbose=0)
                next_state, reward, terminated, truncated, _ = eval_env.step(action)
                next_state = np.reshape(next_state, [1, state_size])

                round_reward += reward
                state = next_state

                if terminated or truncated:
                    eval_reward.append(round_reward)
                    break

        eval_env.close()

        eval_reward_mean = np.sum(eval_reward)/len(eval_reward)
        eval_reward_var = np.var(eval_reward)
        
        return eval_reward_mean, eval_reward_var
    
    def early_stopping(self, ep, eval_reward_mean, threshold=500):
        '''早停机制 如果评估奖励均值达到指定阈值，则提前停止训练 以下为默认样例'''
        if eval_reward_mean >= threshold: # [Modify this threshold as needed]
            print(f"Early stopping triggered at Episode {ep + 1}.")
            return True
        return False
    
    def update(self, action):
        '''执行动作后 更新Agent的状态'''
        state_size = self.env.observation_space.shape[0]
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        done = terminated or truncated
        return next_state, reward, done
    
    def prepare():
        '''在子类方法重写 描述训练之前的所有准备'''
        pass
    
    def update_policy(self, experience, **kwargs):
        return self.policy.update(experience, **kwargs)
    
    def learn(self):
        '''默认中每个 baseline approach 都应该使用相同的 TrainingConfig'''
        # prepare all materials before training
        cfg = self.cfg
        self.prepare()
        score_uplimit = cfg.score_uplimit
        epsilon = cfg.epsilon
        train_counter = 0 # Train Counter for weight syncing
        total_training_time = 0 # For timing training
        state_size = self.env.observation_space.shape[0] # Number of observations (CartPole)
        action_size = self.env.action_space.n            # Number of possible actions
        
        for ep in range(cfg.episode):
            state, _ = self.env.reset()
            state = np.reshape(state, [1, state_size])
            total_reward = 0
            start = time.time() # record start time

            for _ in range(score_uplimit):
                observation = (state, action_size, epsilon)
                action = self.act(observation, deterministic=False, verbose=0)
                next_state, reward, done = self.update(action)
                # zip all into experiences
                experience = (state, action, reward, next_state, done)
                update_dict = {
                    'total_reward': total_reward,
                    'epsilon': epsilon,
                    'train_counter': train_counter,
                    'gamma': cfg.gamma,
                    'ba': cfg.ba,
                    'epsilon_min': cfg.epsilon_min,
                    'epsilon_decay': cfg.epsilon_decay,
                    'target_update_freq': cfg.target_update_freq,
                    'epoch': cfg.epoch,
                }
                # unzip the updating results
                state, total_reward, epsilon, train_counter= self.update_policy(experience, **update_dict)
                if done: 
                    break
                
            # record end time and log training time
            end = time.time()
            total_training_time += end - start

            # Evaluation
            # [WriteCode]   
            eval_reward_mean, eval_reward_var = self.evaluate(max_timesteps=score_uplimit)

            # Log
            self.collect(total_reward, epsilon, eval_reward_mean, eval_reward_var, ep, cfg.episode)

            # Early Stopping Condition to avoid overfitting
            # If the evaluation reward reaches the specified threshold, stop training early.
            # The default threshold is set to 500, but you should adjust this based on observed training performance.
            if self.early_stopping(ep, eval_reward_mean, threshold=score_uplimit):
                break

        # record end time and calculate average training time per episode
        # evaluate average training time per episode
        print(f"Training time: {total_training_time/cfg.episode:.4f} seconds per episode")
        self.env.close()
        
    def save(self, name):
        '''保存训练检查点'''
        self.policy.save(name)
        obj = {
            'train_reward_lst': self.train_reward_lst, 
            'eval_reward_mean_lst': self.eval_reward_mean_lst, 
            'eval_reward_var_lst': self.eval_reward_var_lst
        }
        with open(f'{name}/data.json', 'w', encoding='utf-8') as f:
            json.dump(obj, f)
        f.close()
            
    def load(self, name):
        '''加载训练检查点'''
        policy = self.policy.load(name)
        with open(f'{name}/data.json', 'r', encoding='utf-8') as f:
            obj = json.load(f)
        f.close()
        return policy, obj['train_reward_lst'], obj['eval_reward_mean_lst'], obj['eval_reward_var_lst']
    
    def plot_smoothed_training_rwd(self, window_size=20):
        """Plot smoothed training rewards using a moving average."""
        arr = np.asarray(self.train_reward_lst, dtype=float)
        window_size = max(1, int(window_size))
        if window_size > arr.size:
            window_size = arr.size
        window = np.ones(window_size) / window_size
        smoothed = np.convolve(arr, window, mode='valid')

        x_raw = np.arange(arr.size)
        x_smooth = np.arange(window_size - 1, arr.size)
        
        fig, ax = plt.subplots()
        ax.plot(x_raw, arr, alpha=0.3, label='raw reward')
        ax.plot(x_smooth, smoothed, color='C1', label=f'smoothed (w={window_size})')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Training Reward (raw and smoothed)')
        ax.legend()
        ax.grid(True)
        return fig, ax
    
    def plot_eval_rwd_var(self):
        """Plot evaluation reward variance."""
        arr = np.asarray(self.eval_reward_var_lst, dtype=float)
        x_raw = np.arange(arr.size)
        fig, ax = plt.subplots()
        ax.plot(x_raw, arr, alpha=0.3, label='reward variance')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward Variance')
        ax.set_title('Training Reward Variance')
        ax.legend()
        ax.grid(True)
        return fig, ax