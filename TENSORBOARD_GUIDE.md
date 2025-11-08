# TensorBoard实时监控指南

## 概述

TensorBoard已集成到训练流程中，可以实时监控以下指标：

### 📊 可视化指标

#### 1. **训练指标** (Training/)
- `Episode_Total_Reward` - 每个episode的总奖励
- `Epsilon` - 探索率衰减曲线
- `Batch_Loss` - 每个训练批次的TD损失

#### 2. **评估指标** (Evaluation/)
- `Reward_Mean` - 评估阶段的平均奖励
- `Reward_Variance` - 评估奖励的方差

---

## 🚀 使用方法

### 方法1: 本地运行TensorBoard (推荐用于notebook)

**步骤1**: 在训练开始前启动TensorBoard

```bash
# 在终端中运行
tensorboard --logdir=logs/ --port=6006
```

**步骤2**: 打开浏览器访问
```
http://localhost:6006
```

**步骤3**: 运行训练
```bash
jupyter notebook train_dqn_notebook.ipynb
```

现在你可以在浏览器中实时看到训练进度！

---

### 方法2: 远程服务器使用TensorBoard

如果你在远程GPU服务器上训练：

**在服务器上启动TensorBoard**:
```bash
tensorboard --logdir=logs/ --port=6006 --bind_all
```

**方法A: SSH端口转发**
```bash
# 在本地机器上运行
ssh -L 6006:localhost:6006 user@server_ip

# 然后在本地浏览器访问
http://localhost:6006
```

**方法B: 直接访问服务器IP**
```
http://server_ip:6006
```

---

### 方法3: Jupyter Notebook内嵌TensorBoard

在notebook中添加这个cell：

```python
# 加载TensorBoard扩展
%load_ext tensorboard

# 在notebook内启动TensorBoard
%tensorboard --logdir logs/
```

这样TensorBoard就会直接显示在notebook中！

---

## 📁 日志目录结构

```
logs/
├── dqn_notebook/          # Notebook训练日志
├── optimized_run/         # 优化版本训练日志
└── comparison/            # 对比实验日志
```

每个目录下包含：
```
dqn_notebook/
├── events.out.tfevents.xxx  # TensorBoard事件文件
└── ... (其他训练输出)
```

---

## 🔍 实时监控的关键指标

### 1. Episode_Total_Reward (训练奖励)
- **期望**: 随episode增加而上升
- **正常范围**: 0 → 500 (CartPole-v1)
- **异常**: 如果一直在低值徘徊,检查超参数

### 2. Evaluation/Reward_Mean (评估奖励)
- **期望**: 稳定上升,噪声较小
- **收敛值**: ~500 (CartPole-v1满分)
- **用途**: 更可靠的性能指标

### 3. Batch_Loss (训练损失)
- **期望**: 初期较高,然后逐渐下降并稳定
- **异常**: 如果持续上升或振荡剧烈,可能学习率过大

### 4. Epsilon (探索率)
- **期望**: 从1.0指数衰减到epsilon_min (默认0.01)
- **检查**: 确保衰减速度合理

---

## ⚙️ 高级配置

### 调整日志刷新频率

默认情况下,每个episode结束后写入一次日志。如果你想更频繁地更新：

**修改 `agent/core.py:155`**:
```python
# 每N个episode刷新一次
if ep % 10 == 0:
    self.collect(...)
```

### 添加自定义指标

在 `agent/core.py:collect()` 方法中添加：

```python
if self.cb:
    with tf.summary.create_file_writer(self.cb.log_dir).as_default():
        # 现有指标
        tf.summary.scalar('Training/Episode_Total_Reward', total_reward, step=ep)

        # 添加自定义指标
        tf.summary.scalar('Custom/Average_Episode_Length', avg_length, step=ep)
        tf.summary.scalar('Custom/Learning_Rate', current_lr, step=ep)
```

### 记录直方图和分布

记录Q值分布：

```python
# 在 policy/dqn.py 的 update() 方法中
if cb and train_counter % 100 == 0:  # 每100步记录一次
    with tf.summary.create_file_writer(cb.log_dir).as_default():
        # 记录Q值分布
        q_values = self.model.online_model(states, training=False)
        tf.summary.histogram('Q_Values/Distribution', q_values, step=train_counter)

        # 记录梯度范数
        tf.summary.scalar('Gradients/Norm', tf.linalg.global_norm(grads), step=train_counter)
```

---

## 🎯 使用技巧

### 1. 对比多次实验
```bash
# 运行多个实验
python train.py --logdir=logs/exp1 --lr=0.001
python train.py --logdir=logs/exp2 --lr=0.0001

# TensorBoard同时显示
tensorboard --logdir=logs/
```

TensorBoard会自动显示所有子目录的对比曲线。

### 2. 平滑曲线查看
在TensorBoard界面左侧找到 **Smoothing** 滑块，调整到0.6-0.9可以更清晰地看到趋势。

### 3. 实时性能监控
同时打开两个窗口：
- 窗口1: TensorBoard (查看训练曲线)
- 窗口2: `nvidia-smi -l 1` (查看GPU利用率)

### 4. 下载数据进行后处理
点击TensorBoard右上角的下载按钮，可以导出CSV格式数据用于论文绘图。

---

## 🐛 故障排查

### 问题1: TensorBoard显示"No dashboards are active"
**原因**: 日志目录为空或路径错误

**解决**:
```bash
# 检查日志文件是否存在
ls -la logs/dqn_notebook/

# 确认TensorBoard指向正确目录
tensorboard --logdir=logs/dqn_notebook/
```

### 问题2: 曲线不更新
**原因**: TensorBoard缓存

**解决**:
```bash
# 强制刷新浏览器 (Ctrl+F5)
# 或重启TensorBoard
tensorboard --logdir=logs/ --reload_interval=5
```

### 问题3: "Permission denied"错误
**原因**: 日志目录权限问题

**解决**:
```bash
chmod -R 755 logs/
```

### 问题4: 端口被占用
**原因**: 6006端口已被使用

**解决**:
```bash
# 使用其他端口
tensorboard --logdir=logs/ --port=6007

# 或杀死占用端口的进程
lsof -ti:6006 | xargs kill -9
```

---

## 📚 TensorBoard界面说明

### Scalars (标量)
- 查看损失、奖励等数值指标随时间变化
- 可以选择多个run进行对比
- 支持对数坐标、平滑等

### Graphs (计算图)
- 显示TensorFlow计算图结构
- 查看模型架构和数据流

### Distributions (分布)
- 查看权重、激活值等的分布随时间的变化

### Histograms (直方图)
- 3D视图显示参数分布的演变

### Time Series (时间序列)
- 查看指标的详细时间序列数据

---

## 🎨 最佳实践

### 1. 命名规范
```python
# 使用层次化命名
tf.summary.scalar('Training/DQN/Loss', loss, step=step)
tf.summary.scalar('Training/DDQN/Loss', loss, step=step)
tf.summary.scalar('Evaluation/CartPole/Reward', reward, step=step)
```

这样TensorBoard会自动分组显示。

### 2. 定期清理旧日志
```bash
# 删除7天前的日志
find logs/ -type f -mtime +7 -delete
```

### 3. 使用多个writer
```python
# 训练和验证使用不同的writer
train_writer = tf.summary.create_file_writer('logs/train')
eval_writer = tf.summary.create_file_writer('logs/eval')

with train_writer.as_default():
    tf.summary.scalar('loss', train_loss, step=step)

with eval_writer.as_default():
    tf.summary.scalar('accuracy', eval_acc, step=step)
```

---

## 📊 示例：完整监控设置

```python
# 在notebook中运行完整监控
%load_ext tensorboard
%tensorboard --logdir logs/

# 在另一个cell中训练
from trainer import Trainer
from config.train import TrainingConfig

trainer = Trainer(device='auto', log_dir='logs/dqn_realtime')
train_cfg = TrainingConfig.from_yaml('config/hp_base.yml')
env = gym.make('CartPole-v1')

# 开始训练 - TensorBoard会实时更新
trainer.train_dqn(train_cfg, env, trainer.cb)
```

现在你可以在TensorBoard中实时看到：
- 📈 训练奖励上升
- 📉 损失下降
- 🎯 Epsilon衰减
- ✅ 评估性能提升

---

## 🔗 更多资源

- [TensorBoard官方文档](https://www.tensorflow.org/tensorboard)
- [TensorBoard GitHub](https://github.com/tensorflow/tensorboard)
- [TensorBoard.dev](https://tensorboard.dev/) - 在线分享训练结果

---

**提示**: 训练时保持TensorBoard打开，可以立即发现训练问题并及时调整！
