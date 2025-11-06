import gymnasium as gym
import keras
import numpy as np
import time 
from typing import Any, Callable
import os

from config.train import TrainingConfig
from config.network import NetworkConfig
from policy.base import BasePolicy, DoneException
from agent.core import CoreAgent
from agent.kytolly import KytollyAgent
from policy.ddqn import DoubleDeepQNetworkPolicy
from policy.dqn import DQNPolicy
from model.ddqn_mlp import DoubleDeepQNetworkModel
from model.dqn_mlp import DeepQNetworkModel

class Trainer():
    def __init__(self, device, log_dir,):
        # Set up environment
        self.env = gym.make("CartPole-v1")
        self.state_size = self.env.observation_space.shape[0] # Number of observations (CartPole)
        self.action_size = self.env.action_space.n            # Number of possible actions
        self.cb = keras.callbacks.TensorBoard(log_dir = log_dir, histogram_freq=1)
        self.device = device
    
    def train(self):
        '''默认中每个 baseline approach 都应该使用相同的 TrainingConfig'''
    def train_kytolly(self, train_cfg, env, cb):
        online_ddqn_cfg = NetworkConfig(hidden_dims=[128, 128], metrics=['mse'])
        ddqn_model = DoubleDeepQNetworkModel(online_ddqn_cfg)
        xqy_policy = DoubleDeepQNetworkPolicy(model=ddqn_model,device=self.device)
        xqy_agent = KytollyAgent(env, xqy_policy, train_cfg, cb)
        xqy_agent.learn()
        
        fig0, ax0 = online_ddqn_cfg.table()
        fig1, ax1 = train_cfg.table()
        fig2, ax2 = xqy_agent.plot_smoothed_training_rwd()
        fig3, ax3 = xqy_agent.plot_eval_rwd_var()
        
        fig0.savefig('results/ddqn/8/online_ddqn_cfg.png')
        fig1.savefig('results/ddqn/8/train_cfg.png')
        fig2.savefig('results/ddqn/8/plot_smoothed_training_rwd.png')
        fig3.savefig('results/ddqn/8/plot_eval_rwd_var.png')

    def train_dqn(self, train_cfg, env, cb):
        '''使用标准DQN算法进行训练'''
        dqn_cfg = NetworkConfig(hidden_dims=[128, 128], metrics=['mse'], lr=train_cfg.lr)
        dqn_model = DeepQNetworkModel(dqn_cfg)
        dqn_policy = DQNPolicy(model=dqn_model, device=self.device)
        dqn_agent = KytollyAgent(env, dqn_policy, train_cfg, cb)
        dqn_agent.learn()

        fig0, ax0 = dqn_cfg.table()
        fig1, ax1 = train_cfg.table()
        fig2, ax2 = dqn_agent.plot_smoothed_training_rwd()
        fig3, ax3 = dqn_agent.plot_eval_rwd_var()

        # 创建results/dqn目录
        os.makedirs('results/dqn/8', exist_ok=True)

        fig0.savefig('results/dqn/8/dqn_cfg.png')
        fig1.savefig('results/dqn/8/train_cfg.png')
        fig2.savefig('results/dqn/8/plot_smoothed_training_rwd.png')
        fig3.savefig('results/dqn/8/plot_eval_rwd_var.png')
        