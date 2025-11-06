# policy/q_learning.py
from policy.base import BasePolicy
import numpy as np

class QNetworkPolicy(BasePolicy):
    def __init__(self, model, optimizer, device=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def select_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return np.random.choice(self.model.output_shape[1])  # action_size
        else:
            q_values = self.model(state, training=False).numpy()
            return np.argmax(q_values[0])