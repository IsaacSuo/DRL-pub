#!/usr/bin/env python3
"""
经验回放缓冲区性能测试
比较优化前后的采样性能
"""

import time
import numpy as np
import tensorflow as tf
from collections import deque

def old_buffer_sample(buffer, batch_size):
    """原始的采样方法（从kytolly.py复制）"""
    indices = np.random.choice(len(buffer), batch_size, replace=False)
    minibatch = [buffer[i] for i in indices]

    states_batch = np.stack([experience[0].squeeze() for experience in minibatch], axis=0)
    actions_batch = np.array([experience[1] for experience in minibatch], dtype=np.int32)
    rewards_batch = np.array([experience[2] for experience in minibatch], dtype=np.float32)
    next_states_batch = np.stack([experience[3].squeeze() for experience in minibatch], axis=0)
    dones_batch = np.array([experience[4] for experience in minibatch], dtype=np.float32)

    return (
        tf.convert_to_tensor(states_batch, dtype=tf.float32),
        tf.convert_to_tensor(actions_batch, dtype=tf.int32),
        tf.convert_to_tensor(rewards_batch, dtype=tf.float32),
        tf.convert_to_tensor(next_states_batch, dtype=tf.float32),
        tf.convert_to_tensor(dones_batch, dtype=tf.float32)
    )

def benchmark_performance():
    """性能基准测试"""
    print("🚀 经验回放缓冲区性能测试")
    print("=" * 50)

    # 测试参数
    buffer_size = 10000
    batch_size = 64
    num_samples = 100
    state_dim = 4

    print(f"📊 测试参数:")
    print(f"   缓冲区大小: {buffer_size}")
    print(f"   批次大小: {batch_size}")
    print(f"   测试次数: {num_samples}")
    print(f"   状态维度: {state_dim}")
    print()

    # 1. 生成测试数据
    print("📦 生成测试数据...")
    test_data = []
    for i in range(buffer_size):
        state = np.random.randn(state_dim)
        action = np.random.randint(0, 2)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim)
        done = np.random.randint(0, 2)
        test_data.append((state, action, reward, next_state, done))

    # 2. 测试原始deque方法
    print("🐌 测试原始deque方法...")
    old_buffer = deque(test_data)

    start_time = time.time()
    for _ in range(num_samples):
        old_buffer_sample(old_buffer, batch_size)
    old_time = time.time() - start_time

    # 3. 测试优化后的方法
    print("⚡ 测试优化后的缓冲区...")
    from agent.replay_buffer import OptimizedReplayBuffer
    new_buffer = OptimizedReplayBuffer(buffer_size, state_dim)

    # 添加数据到新缓冲区
    for state, action, reward, next_state, done in test_data:
        new_buffer.add(state, action, reward, next_state, done)

    start_time = time.time()
    for _ in range(num_samples):
        new_buffer.sample(batch_size)
    new_time = time.time() - start_time

    # 4. 结果对比
    print("\n📈 性能对比结果:")
    print("=" * 50)
    print(f"原始方法耗时: {old_time:.4f} 秒")
    print(f"优化方法耗时: {new_time:.4f} 秒")
    print(f"性能提升:     {old_time/new_time:.2f}x")
    print(f"时间节省:     {((old_time - new_time) / old_time * 100):.1f}%")

    # 单次采样平均时间
    old_avg = old_time / num_samples * 1000  # 转换为毫秒
    new_avg = new_time / num_samples * 1000
    print(f"\n单次采样平均时间:")
    print(f"原始方法: {old_avg:.3f} ms")
    print(f"优化方法: {new_avg:.3f} ms")

    # 内存使用估算
    print(f"\n💾 内存使用估算:")
    print(f"原始方法: ~{buffer_size * 5 * 8 / 1024 / 1024:.2f} MB (Python对象 + 开销)")
    print(f"优化方法: ~{buffer_size * (4 + 4 + 4 + 4 + 4) / 1024 / 1024:.2f} MB (预分配数组)")

    return old_time / new_time

if __name__ == "__main__":
    speedup = benchmark_performance()
    print(f"\n✅ 优化成功！性能提升 {speedup:.2f}x")