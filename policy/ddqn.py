from base import BasePolicy
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from collections import deque
from gymnasium import Env

from model.ddqn import DoubleDeepQNetworkTagetModel, DoubleDeepQNetworkEvalModel

class DoubleDeepQNetworkPolicy(BasePolicy):
    def __init__(self, 
                 target_model: DoubleDeepQNetworkTagetModel,
                 eval_model: DoubleDeepQNetworkEvalModel):
        super().__init__(self)
        self.taeget_model = target_model
        self.eval_model = eval_model
        
    def prepare(self):
        self.init_buffer()
        
    def init_buffer(self):
        '''
        构建经验回放缓冲区‘
        '''
        self.replay_buffer = deque(maxlen=10000)
        
    def store_experience(self, state, action, reward, next_state, done):
        '''
        存储单个经验到回放缓冲区
        '''
        self.replay_buffer.append((state, action, reward, next_state, done))
        
    def clear_buffer(self):
        '''
        清空经验回放缓冲区
        '''
    
    def step(self, env: Env, state, action, state_size, total_reward, train_counter):
        '''
        与环境交互一步 即执行一步训练，包括前向传播、计算损失、反向传播等。
        '''
        state, action, reward, next_state, done = super().step(env, state, action, state_size, total_reward)
        # store experience into replay buffer
        # [WriteCode]
        self.store_experience(state, action, reward, next_state, done)
        # if len(replay_buffer) >= ba:
        #   train_counter += 1
            # Update policy with mini-batches if replay buffer contains enough samples
            # Update policy using Double Deep Q-Learning update:
            # Q(s, a) = r + gamma * Q_target(S', argmax Q_eval(S', a))
            # [WriteCode]

            # Compute target Q-values:
            # - If done, Q-target = reward (no future reward)
            # - Otherwise, Q-target = reward + gamma * Q_target(S', argmax Q_eval(S', a))

            # Predict current Q-values for state using eval_model
            # Use eval_model to determine best action in next_state
            # Use target_model to compute Q-value for that action

            # Update only the Q-value for the taken action

            # Fit the model:
            # - Inputs: state
            # - Targets: updated Q-values (with action Q-value replaced by computed target)

            # Update exploration rate
            # if epsilon > epsilon_min:
            #     epsilon *= epsilon_decay

            # Periodically update the target network
            # if train_counter % target_update_freq == 0:
            #     target_model.set_weights(eval_model.get_weights())
    
    def evaluate(self, ep, episode, total_reward, epsilon):
        '''
        评估当前策略的表现
        '''
        print(f"Episode {ep + 1}/{episode} | Ep. Total Reward: {total_reward}"
        f" | Epsilon : {epsilon:.3f}")
#        f" | Eval Rwd Mean: {eval_reward_mean:.2f}"
#        f" | Eval Rwd Var: {eval_reward_var:.2f}")