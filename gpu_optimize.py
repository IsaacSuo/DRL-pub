"""
GPU性能优化配置脚本
在训练脚本开头导入此模块以启用所有GPU优化
Usage:
    from gpu_optimize import configure_gpu_optimization
    configure_gpu_optimization()
"""

import os
import tensorflow as tf


def configure_gpu_optimization(
    enable_xla: bool = True,
    enable_mixed_precision: bool = False,
    memory_growth: bool = True,
    verbose: bool = True
):
    """
    配置TensorFlow的GPU优化选项

    Args:
        enable_xla: 启用XLA(加速线性代数)编译优化 (推荐: True)
        enable_mixed_precision: 启用混合精度训练 (需要Tensor Core支持)
        memory_growth: 启用GPU内存按需增长,防止OOM (推荐: True)
        verbose: 打印优化配置信息

    Returns:
        dict: 优化配置状态
    """

    optimization_status = {}

    # 1. XLA编译优化
    if enable_xla:
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
        tf.config.optimizer.set_jit(True)
        optimization_status['XLA'] = '✅ Enabled'
        if verbose:
            print("✅ XLA compilation enabled (expected 5-15% speedup)")
    else:
        optimization_status['XLA'] = '❌ Disabled'

    # 2. GPU内存增长
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        optimization_status['GPUs_found'] = len(gpus)

        if memory_growth:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                optimization_status['Memory_growth'] = '✅ Enabled'
                if verbose:
                    print(f"✅ Configured {len(gpus)} GPU(s) with memory growth enabled")
            except RuntimeError as e:
                optimization_status['Memory_growth'] = f'⚠️  Warning: {e}'
                if verbose:
                    print(f"⚠️  GPU memory growth warning: {e}")
        else:
            optimization_status['Memory_growth'] = '❌ Disabled'
    else:
        optimization_status['GPUs_found'] = 0
        optimization_status['Device'] = 'CPU'
        if verbose:
            print("⚠️  No GPU detected, running on CPU")

    # 3. 混合精度训练
    if enable_mixed_precision:
        try:
            from tensorflow.keras import mixed_precision
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            optimization_status['Mixed_precision'] = '✅ Enabled (float16)'
            if verbose:
                print("✅ Mixed precision training enabled (float16)")
                print("   Note: Requires Tensor Cores for optimal performance")
        except Exception as e:
            optimization_status['Mixed_precision'] = f'❌ Failed: {e}'
            if verbose:
                print(f"⚠️  Mixed precision failed: {e}")
    else:
        optimization_status['Mixed_precision'] = '❌ Disabled'

    # 4. 其他性能优化
    # 启用TensorFlow的确定性操作(可选,可能略微降低性能)
    # os.environ['TF_DETERMINISTIC_OPS'] = '1'

    # 禁用不必要的日志
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # 线程优化
    tf.config.threading.set_inter_op_parallelism_threads(0)  # 自动
    tf.config.threading.set_intra_op_parallelism_threads(0)  # 自动

    optimization_status['Threading'] = 'Auto-configured'

    if verbose:
        print("\n" + "="*60)
        print("GPU Optimization Summary:")
        print("="*60)
        for key, value in optimization_status.items():
            print(f"  {key:20s}: {value}")
        print("="*60 + "\n")

    return optimization_status


def benchmark_gpu_performance(iterations: int = 100, batch_size: int = 32):
    """
    简单的GPU性能基准测试

    Args:
        iterations: 测试迭代次数
        batch_size: 批次大小

    Returns:
        float: 平均每次迭代时间(毫秒)
    """
    import time
    import numpy as np

    print(f"\n🔍 Running GPU benchmark ({iterations} iterations, batch_size={batch_size})...")

    # 创建测试数据
    x = tf.random.normal([batch_size, 4])
    y = tf.random.normal([batch_size, 2])

    # 简单的测试模型
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = tf.keras.losses.MeanSquaredError()

    @tf.function(jit_compile=True)
    def train_step(x, y):
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            loss = loss_fn(y, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    # 预热
    for _ in range(10):
        train_step(x, y)

    # 基准测试
    start_time = time.time()
    for _ in range(iterations):
        train_step(x, y)
    end_time = time.time()

    avg_time_ms = (end_time - start_time) / iterations * 1000

    print(f"✅ Benchmark complete: {avg_time_ms:.3f} ms/iteration")
    print(f"   Throughput: {1000/avg_time_ms:.1f} iterations/second\n")

    return avg_time_ms


if __name__ == "__main__":
    # 运行优化配置
    print("="*60)
    print("DRL GPU Optimization Configuration")
    print("="*60 + "\n")

    status = configure_gpu_optimization(
        enable_xla=True,
        enable_mixed_precision=False,  # 默认关闭,需要手动启用
        memory_growth=True,
        verbose=True
    )

    # 运行基准测试
    if status.get('GPUs_found', 0) > 0:
        benchmark_gpu_performance(iterations=100, batch_size=32)
    else:
        print("⚠️  Skipping benchmark (no GPU available)")
