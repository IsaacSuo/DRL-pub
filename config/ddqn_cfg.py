from keras.losses import Loss

class DDQNConfig:
    def __init__(self,
                 input_dim:int,
                 output_dim: int,
                 use_bias: bool,
                 is_compiled: bool,
                 hidden_dims: list,
                 optimizer: str,
                 lr: str,
                 loss: Loss,
                 metrics:list):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.is_compiled = is_compiled
        self.hidden_dims = hidden_dims
        self.optimizer = optimizer
        self.lr = lr
        self.loss = loss
        self.metrics = metrics