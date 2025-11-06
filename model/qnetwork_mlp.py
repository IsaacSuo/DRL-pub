# model/q_network_mlp.py
from keras.models import Sequential
from keras.layers import Dense, ReLU
from config.qnetwork_cfg import QNetworkConfig


class QNetworkModel:
    def __init__(self, config: QNetworkConfig):
        self.config = config
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        # Input layer
        model.add(Dense(
            self.config.hidden_dims[0],
            input_dim=self.config.input_dim,
            use_bias=self.config.use_bias
        ))
        model.add(ReLU())
        # Hidden layers
        for units in self.config.hidden_dims[1:]:
            model.add(Dense(units, use_bias=self.config.use_bias))
            model.add(ReLU())
        # Output layer
        model.add(Dense(self.config.output_dim, use_bias=self.config.use_bias))
        # Compile
        from keras.optimizers import Adam
        optimizer = Adam(learning_rate=self.config.lr, clipnorm=self.config.clipnorm)
        model.compile(optimizer=optimizer, loss=self.config.loss, metrics=self.config.metrics)
        return model

    def __call__(self, *args, **kwargs):
        # 使 model(state) 等价于 model.model(state)
        return self.model(*args, **kwargs)

    # 可选：代理属性，方便访问
    @property
    def trainable_variables(self):
        return self.model.trainable_variables

    @property
    def output_shape(self):
        return self.model.output_shape