import numpy as np
import tensorflow as tf
from typing import Tuple

class OptimizedReplayBuffer:
    """
    高性能经验回放缓冲区
    使用预分配的numpy数组实现快速采样
    """

    def __init__(self, max_size: int, state_dim: int):
        self.max_size = max_size
        self.state_dim = state_dim

        # 预分配所有存储空间
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros(max_size, dtype=np.int32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=np.float32)

        # 环形缓冲区管理
        self.position = 0
        self.size = 0

    def add(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: float):
        """添加单个经验到缓冲区"""
        # 确保状态是1D的
        state = state.squeeze() if hasattr(state, 'squeeze') else state
        next_state = next_state.squeeze() if hasattr(next_state, 'squeeze') else next_state

        # 存储到当前位置
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done

        # 更新位置和大小
        self.position = (self.position + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor,
                                               tf.Tensor, tf.Tensor]:
        """
        高度优化的批量采样
        直接通过numpy索引一次性获取所有数据
        """
        if self.size < batch_size:
            raise ValueError(f"Buffer has only {self.size} samples, but {batch_size} requested")

        # 随机选择索引
        indices = np.random.choice(self.size, batch_size, replace=False)

        # 一次性向量化采样所有组件
        states_batch = self.states[indices]
        actions_batch = self.actions[indices]
        rewards_batch = self.rewards[indices]
        next_states_batch = self.next_states[indices]
        dones_batch = self.dones[indices]

        # 一次性转换为TensorFlow张量
        return (
            tf.convert_to_tensor(states_batch, dtype=tf.float32),
            tf.convert_to_tensor(actions_batch, dtype=tf.int32),
            tf.convert_to_tensor(rewards_batch, dtype=tf.float32),
            tf.convert_to_tensor(next_states_batch, dtype=tf.float32),
            tf.convert_to_tensor(dones_batch, dtype=tf.float32)
        )

    def __len__(self) -> int:
        return self.size