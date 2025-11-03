import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Dense, BatchNormalization
from config.ddqn_cfg import DDQNConfig
import numpy as np
class DoubleDeepQNetworkTagetModel():
    def __init__(self, cfg: DDQNConfig):
        self.input = Input(shape=(cfg.input_dim, ))
        self.model = Sequential()
        # [WriteCode]
        input_dim = cfg.input_dim
        for hd in cfg.hidden_dims:
            self.model.add(Dense(hd, input_shape=(input_dim,), activation='relu', use_bias=True))
            self.model.add(BatchNormalization())
            input_dim = hd
        self.model.add(Dense(cfg.output_dim, activation='sigmoid'))

        # Compile the model
        if cfg.is_compiled:
            self.model.compile(optimizer=cfg.optimizer, loss=cfg.loss, metrics=cfg.metrics)
    
    def summary(self):
        return self.model.summary()
    
    def predict(self, state, verbose=0):
        '''
        给定环境状态 输出 qvalues
        '''
        return self.model.predict(state, verbose=verbose)
    
    def compile(self):
        '''
        不应该实现 target_model 是 eval_model 的 periodic copy
        '''
        pass
    
    def set_weights(self, weights):
        '''
        设置模型权重 应该保持和 eval_model 一致
        '''
        self.model.set_weights(weights)
    
class DoubleDeepQNetworkOnlineModel():
    def __init__(self, cfg: DDQNConfig):
        self.input = Input(shape=(cfg.input_dim, ))
        self.model = Sequential()
        # [WriteCode]
        input_dim = cfg.input_dim
        for hd in cfg.hidden_dims:
            self.model.add(Dense(hd, input_shape=(input_dim,), activation='relu', use_bias=True))
            self.model.add(BatchNormalization())
            input_dim = hd
        self.model.add(Dense(cfg.output_dim, activation='sigmoid'))

        # Compile the model
        if cfg.is_compiled:
            self.compile(optimizer=cfg.optimizer,loss=cfg.loss, metrics=cfg.metrics)
    
    def predict(self, state, verbose=0):
        '''
        给定环境状态 输出 qvalues
        '''
        return self.model.predict(state, verbose=verbose)
    
    def summary(self):
        return self.model.summary()
    
    def compile(self, optimizer, loss, metrics):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    def get_weights(self):
        '''
        获取模型权重
        '''
        return self.model.get_weights()
    
    def fit(self, X, Y, batch_size=None, callbacks=None, verbose=1, epoch=1):
        return self.model.fit(X, Y, batch_size=batch_size, callbacks=callbacks, verbose=verbose, epochs=epoch)