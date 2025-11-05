import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard
from config.train import TrainingConfig
import numpy as np
import gymnasium as gym
from gymnasium import Env

class DoneException(Exception):
    def __init__(self):
        self.message = 'get DONE signal, this interaction step is done.'

    def __str__(self):
        return f"{self.message}"
    
class BasePolicy():
    def __init__(self, model: keras.Model, device='auto'):
        self.device = device
        self.model = model
    def get_action(self, state, verbose=0):
        '''
        模型的预测模块，给定当前观察到的 agent 的状态 输出动作概率分布
        state: 当前观察到的 agent 的状态
        verbose: 是否打印日志信息，默认为 0 不打印
        '''
        # Force verbose=0 for silent predictions
        return self.model.predict(state, verbose=0)

    def update(self, experiences, **kwargs):
        """使用经验数据更新策略参数"""
        pass
    
    def save(self, path: str):
        """保存策略参数"""
        pass
    
    def load(self, path: str):
        """加载策略参数"""
        pass
    
    def to(self, device):
        """移动模型到指定设备"""
        pass
    
    def train(self):
        """设置为训练模式"""
        pass
    
    def eval(self):
        """设置为评估模式"""
        pass