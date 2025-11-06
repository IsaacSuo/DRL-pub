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
from keras.losses import mean_squared_error, mean_absolute_error
import numpy as np

from config.network import NetworkConfig as DQNConfig

OPTIMIZER_DICT = {
    'Adam': Adam,
    'Adadelta': Adadelta,
    'Adamax': Adamax,
    'AdamW': AdamW,
    'Ftrl': Ftrl,
    'Lion': Lion,
    'Nadam': Nadam,
    'RMSprop': RMSprop,
    'SGD': SGD,
}

LOSSFUNCTION_DICT = {
    'mse': mean_squared_error,
    'mae': mean_absolute_error,
}

class DeepQNetworkModel(keras.Model):
    """
    DQN专用神经网络模型
    包含online和target两个网络，算法逻辑：Q(s,a) = r + γ * max Q_target(s',a')
    """
    def __init__(self, cfg: DQNConfig):
        super().__init__()
        self.online_model: Sequential = Sequential()
        self.target_model: Sequential = Sequential()

        # 创建优化器和损失函数
        custom_optimizer = OPTIMIZER_DICT[cfg.optimizer](learning_rate=cfg.lr, clipnorm=cfg.clipnorm)
        custom_loss_function = LOSSFUNCTION_DICT[cfg.loss]

        self.opt = custom_optimizer
        self.loss_fn = custom_loss_function

        # 构建网络
        self._build(model=self.online_model, cfg=cfg)
        self._build(model=self.target_model, cfg=cfg)

        # 编译online模型（构建网络之后）
        self.online_model.compile(optimizer=custom_optimizer, loss=custom_loss_function, metrics=cfg.metrics)

        # 初始化时同步权重
        self.sync()

    def _build(self, model: Sequential, cfg: DQNConfig):
        """构建神经网络"""
        is_first_layer = True
        for hd in cfg.hidden_dims:
            if is_first_layer:
                model.add(Dense(hd, input_shape=(cfg.input_dim,), use_bias=True))
                is_first_layer = False
            else:
                model.add(Dense(hd, use_bias=True))

            model.add(Activation('relu'))

        # 输出层
        model.add(Dense(cfg.output_dim, activation='linear'))

    def predict(self, state, verbose=0):
        """给定环境状态，输出Q值"""
        return self.online_model.predict(state, verbose=verbose)

    def summary(self):
        """显示模型结构"""
        print("=== Online Model ===")
        self.online_model.summary()
        print("\n=== Target Model ===")
        self.target_model.summary()

    def sync(self):
        """将online模型的权重同步到target模型"""
        self.target_model.set_weights(self.online_model.get_weights())
        print("✅ Target network synchronized with online network")

    def save(self, path):
        """保存模型权重"""
        self.online_model.save_weights(f"{path}_online.h5")
        self.target_model.save_weights(f"{path}_target.h5")
        print(f"✅ Model saved to {path}")

    def load(self, path):
        """加载模型权重"""
        try:
            self.online_model.load_weights(f"{path}_online.h5")
            self.target_model.load_weights(f"{path}_target.h5")
            print(f"✅ Model loaded from {path}")
        except:
            print(f"❌ Failed to load model from {path}")