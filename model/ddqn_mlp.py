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
class DoubleDeepQNetworkModel(keras.Model):
    def __init__(self, cfg: DDQNConfig):
        super().__init__()
        self.online_model: Sequential = Sequential()
        self.target_model: Sequential = Sequential()
        custom_optimizer = OPTIMIZER_DICT[cfg.optimizer](learning_rate=cfg.lr, clipnorm=cfg.clipnorm)
        self.online_model.compile(optimizer=custom_optimizer, loss=cfg.loss, metrics=cfg.metrics)
        self._build(model=self.online_model, cfg=cfg)
        self._build(model=self.target_model, cfg=cfg)
        self.sync()
        
    def _build(self, model: Sequential, cfg: DDQNConfig):
        is_first_layer = True
        for hd in cfg.hidden_dims:
            if is_first_layer:
                model.add(Dense(hd, input_shape=(cfg.input_dim,), use_bias=True))
                is_first_layer = False
            else:
                model.add(Dense(hd, use_bias=True))

            # self.model.add(BatchNormalization())
            model.add(Activation('relu'))
        model.add(Dense(cfg.output_dim, activation='linear'))
    def predict(self, state, verbose=0):
        '''给定环境状态 输出 qvalues'''
        return self.online_model.predict(state, verbose=0)
    
    def summary(self):
        return self.online_model.summary()
    
    def sync(self):
        self.target_model.set_weights(self.online_model.get_weights())
        
    def save(self, path):
        '''保存模型权重'''
        
    def load(self, path):
        '''加载模型权重'''
        