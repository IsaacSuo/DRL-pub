import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from config.train_cfg import TrainingConfig
import numpy as np
import gymnasium as gym
from gymnasium import Env

class DoneException(Exception):
    def __init__(self):
        self.message = 'get DONE signal, this interaction step is done.'

    def __str__(self):
        return f"{self.message}"
    
class BasePolicy():
    def __init__(self):
        self.model = None # 需要在子类中定义具体的模型结构
        
        # For logging
        self.train_reward_lst = []
        self.eval_reward_mean_lst = []
        self.eval_reward_var_lst = []
        
    def predict(self, state, verbose=0):
        '''
        模型的预测模块，给定当前观察到的 agent 的状态 输出动作概率分布
        params:
            state: 当前观察到的 agent 的状态
            verbose: 是否打印日志信息，默认为 0 不打印
        '''
        self.model.predict(state, verbose=verbose)
    
    def __repr__(self):
        '''
        模型的打印信息 summary
        print(...) 会调用该函数
        '''
        return "BasePolicy"
    
    def prepare(self):
        '''
        在训练开始前进行的准备工作，例如初始化经验回放缓冲区等。
        在子类方法下重写该方法时，子类应调用 super().prepare() 以确保基础准备工作被正确执行。
        '''
        pass
        
    def act(self, state, action_size, epsilon, versbose=0):
        '''
        根据状态选择动作
        示例采用 epsilon-greedy policy
        '''
        if np.random.rand() <= epsilon:
            action = np.random.choice(action_size)
        else:
            # pass # remove pass and use 2 lines below
            q_values = self.predict(state, verbose=0)
            action = np.argmax(q_values)
        return action
        
    def step(self, env: Env, action, state_size):
        '''
        正确更新 state, total_reward, epsilon
        在子类方法中重写
        '''
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        done = terminated or truncated
        return next_state, reward, done

    def evaluate(self, model: keras.Model, max_timesteps=500):
        '''
        评估当前策略的表现
        '''
        eval_env = gym.make("CartPole-v1")
        state_size = eval_env.observation_space.shape[0] # Number of observations (CartPole)
        action_size = eval_env.action_space.n            # Number of possible actions
        eval_reward = []

        for i in range (5):
            round_reward = 0
            state, _ = eval_env.reset()
            state = np.reshape(state, [1, state_size])

            for i in range(max_timesteps):
                action = np.argmax(model.predict(state, verbose=0)[0])
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
    
    def log(self, total_reward, eval_reward_mean, eval_reward_var, ep, episode, epsilon):
        self.train_reward_lst.append(total_reward)
        self.eval_reward_mean_lst.append(eval_reward_mean)
        self.eval_reward_var_lst.append(eval_reward_var)
        print(f"Episode {ep + 1}/{episode} | Ep. Total Reward: {total_reward}| Epsilon : {epsilon:.3f}| Eval Rwd Mean: {eval_reward_mean:.2f}| Eval Rwd Var: {eval_reward_var:.2f}")
        
    def early_stopping(self, ep, eval_reward_mean, threshold=200):
        '''
        早停机制
        如果评估奖励均值达到指定阈值，则提前停止训练
        在每个子类策略中进行重写 以下为默认样例
        '''
        if eval_reward_mean > threshold: # [Modify this threshold as needed]
            print(f"Early stopping triggered at Episode {ep + 1}.")
            return True
        return False