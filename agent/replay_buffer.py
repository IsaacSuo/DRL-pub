import numpy as np
import tensorflow as tf
from typing import Tuple

class OptimizedReplayBuffer:
    """
    GPU加速的高性能经验回放缓冲区
    使用TensorFlow变量在GPU上预分配内存,避免CPU-GPU数据传输瓶颈
    """

    def __init__(self, max_size: int, state_dim: int, device: str = 'auto'):
        self.max_size = max_size
        self.state_dim = state_dim

        # 自动检测设备
        if device == 'auto':
            gpus = tf.config.list_physical_devices('GPU')
            self.device = '/GPU:0' if gpus else '/CPU:0'
        else:
            self.device = device

        # 在指定设备上预分配所有存储空间(使用TensorFlow变量)
        with tf.device(self.device):
            self.states = tf.Variable(
                tf.zeros((max_size, state_dim), dtype=tf.float32),
                trainable=False,
                name='replay_states'
            )
            self.actions = tf.Variable(
                tf.zeros(max_size, dtype=tf.int32),
                trainable=False,
                name='replay_actions'
            )
            self.rewards = tf.Variable(
                tf.zeros(max_size, dtype=tf.float32),
                trainable=False,
                name='replay_rewards'
            )
            self.next_states = tf.Variable(
                tf.zeros((max_size, state_dim), dtype=tf.float32),
                trainable=False,
                name='replay_next_states'
            )
            self.dones = tf.Variable(
                tf.zeros(max_size, dtype=tf.float32),
                trainable=False,
                name='replay_dones'
            )

            # 环形缓冲区管理变量
            self.position = tf.Variable(0, dtype=tf.int32, trainable=False, name='position')
            self.size = tf.Variable(0, dtype=tf.int32, trainable=False, name='size')

        print(f"✅ GPU-accelerated replay buffer initialized on {self.device}")

    def add(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: float):
        """添加单个经验到缓冲区 - 优化版本,减少数据传输"""
        # 确保状态是1D的
        state = state.squeeze() if hasattr(state, 'squeeze') else state
        next_state = next_state.squeeze() if hasattr(next_state, 'squeeze') else next_state

        # 转换为张量(只进行一次CPU->GPU传输)
        with tf.device(self.device):
            idx = self.position.numpy()

            # 使用scatter_nd_update进行原地更新(避免创建副本)
            self.states[idx].assign(state)
            self.actions[idx].assign(action)
            self.rewards[idx].assign(reward)
            self.next_states[idx].assign(next_state)
            self.dones[idx].assign(done)

            # 更新位置和大小
            new_position = (idx + 1) % self.max_size
            new_size = min(self.size.numpy() + 1, self.max_size)

            self.position.assign(new_position)
            self.size.assign(new_size)

    @tf.function(reduce_retracing=True)
    def sample(self, batch_size: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor,
                                               tf.Tensor, tf.Tensor]:
        """
        GPU加速的批量采样 - 全部操作在GPU上完成,零CPU-GPU传输
        """
        # 在GPU上生成随机索引
        indices = tf.random.uniform(
            shape=[batch_size],
            minval=0,
            maxval=self.size,
            dtype=tf.int32
        )

        # 在GPU上直接gather数据(零拷贝)
        states_batch = tf.gather(self.states, indices)
        actions_batch = tf.gather(self.actions, indices)
        rewards_batch = tf.gather(self.rewards, indices)
        next_states_batch = tf.gather(self.next_states, indices)
        dones_batch = tf.gather(self.dones, indices)

        return (
            states_batch,
            actions_batch,
            rewards_batch,
            next_states_batch,
            dones_batch
        )

    def __len__(self) -> int:
        return self.size.numpy()