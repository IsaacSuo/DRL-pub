# -*- coding: utf-8 -*-
import tensorflow as tf
import keras
from keras import Sequential
from keras.layers import Input, Dense, BatchNormalization, Activation
from keras.models import load_model
from keras.optimizers import (
    Adadelta, 
    Adam, 
    Adamax, 
    Ftrl,
    Nadam,
    RMSprop,
    SGD,
)
from keras.losses import (
    BinaryCrossentropy,
    BinaryFocalCrossentropy,
    CategoricalCrossentropy,
    CategoricalHinge,
    CosineSimilarity,
    Hinge,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
import numpy as np

from config.network import NetworkConfig

OPTIMIZER_DICT = {
    'Adam': Adam,
    'Adadelta': Adadelta, 
    'Adam': Adam, 
    'Adamax': Adamax, 
    'Ftrl': Ftrl,
    'Nadam': Nadam,
    'RMSprop': RMSprop,
    'SGD': SGD,
}
LOSSFUNCTION_DICT = {
    'mse': mean_squared_error,
    'mae': mean_absolute_error,
    'mape': mean_absolute_percentage_error,
    'bc': BinaryCrossentropy,
    'bfc':BinaryFocalCrossentropy,
    'cc':CategoricalCrossentropy,
    'ch':CategoricalHinge,
    'cs':CosineSimilarity,
    'h':Hinge,
}
class DoubleDeepQNetworkModel(keras.Model):
    def __init__(self, cfg: NetworkConfig):
        super().__init__()
        self.online_model: Sequential = Sequential()
        self.target_model: Sequential = Sequential()
        custom_optimizer = OPTIMIZER_DICT[cfg.optimizer](learning_rate=cfg.lr, clipnorm=cfg.clipnorm)
        custom_loss_function = LOSSFUNCTION_DICT[cfg.loss]
        self.opt = custom_optimizer
        self.loss_fn = custom_loss_function
        self.online_model.compile(optimizer=custom_optimizer, loss=custom_loss_function, metrics=cfg.metrics)
        self._build(model=self.online_model, cfg=cfg)
        self._build(model=self.target_model, cfg=cfg)
        self.sync()
        
    def _build(self, model: Sequential, cfg: NetworkConfig):
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
    
    @tf.function
    def sync(self):
        """将online模型的权重同步到target模型 - GPU优化版本,无CPU传输"""
        for target_var, online_var in zip(
            self.target_model.trainable_variables,
            self.online_model.trainable_variables
        ):
            target_var.assign(online_var)  # 纯GPU内存操作

    def sync_verbose(self):
        """带打印信息的同步版本,仅在需要时调用"""
        self.sync()
        print("✅ Target network synchronized with online network")
        
    def save(self, name):
        '''保存模型权重'''
        self.online_model.save(f'{name}/online_sequential.h5')
        self.target_model.save(f'{name}/target_sequential.h5')
        
    def load(self, name):
        '''加载模型权重'''
        online_model = load_model(f'{name}/online_sequential.h5')
        target_model = load_model(f'{name}/target_sequential.h5')
        return online_model
        
    def plot(self):
        pass
