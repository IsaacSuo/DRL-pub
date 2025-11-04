import tensorflow as tf
import keras
from keras import Sequential
from keras.layers import Input, Dense, BatchNormalization, Activation
from keras.optimizers import (
    Adadelta, 
    Adam, 
    Adamax, 
    AdamW, 
    Ftrl,
    Lion,
    Nadam,
    RMSprop,
    SGD,
)
import numpy as np

from config.ddqn_cfg import DDQNConfig

OPTIMIZER_DICT = {
    'Adam': Adam,
    'Adadelta': Adadelta, 
    'Adam': Adam, 
    'Adamax': Adamax, 
    'AdamW': AdamW, 
    'Ftrl': Ftrl,
    'Lion': Lion,
    'Nadam': Nadam,
    'RMSprop': RMSprop,
    'SGD': SGD,
}
class DoubleDeepQNetworkTagetModel():
    def __init__(self, cfg: DDQNConfig):
        self.model: keras.Model = Sequential()
        # [WriteCode]
        is_first_layer = True
        for hd in cfg.hidden_dims:
            if is_first_layer:
                self.model.add(Dense(hd, input_shape=(cfg.input_dim,), use_bias=True))
                is_first_layer = False
            else:
                self.model.add(Dense(hd, use_bias=True))

            self.model.add(BatchNormalization())
            self.model.add(Activation('relu'))
        self.model.add(Dense(cfg.output_dim, activation='linear'))

        # Compile the model
        if cfg.is_compiled:
            custom_optimizer = Adam(learning_rate=cfg.lr, clipnorm=cfg.clipnorm)
            self.model.compile(optimizer=custom_optimizer, loss=cfg.loss, metrics=cfg.metrics)
    
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
        self.model: keras.Model = Sequential()
        # [WriteCode]
        is_first_layer = True
        for hd in cfg.hidden_dims:
            if is_first_layer:
                self.model.add(Dense(hd, input_shape=(cfg.input_dim,), use_bias=True))
                is_first_layer = False
            else:
                self.model.add(Dense(hd, use_bias=True))

            self.model.add(BatchNormalization())
            self.model.add(Activation('relu'))
        self.model.add(Dense(cfg.output_dim, activation='linear'))

        # Compile the model
        if cfg.is_compiled:
            custom_optimizer = Adam(learning_rate=cfg.lr, clipnorm=cfg.clipnorm)
            self.model.compile(optimizer=custom_optimizer, loss=cfg.loss, metrics=cfg.metrics)
    
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