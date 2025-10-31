# 🎓 EEC4400：Exploring Reinforcement Learning with Q-Learning, Naïve DQN, DQN and DDQN for Cart-Pole

## 📘 项目简介

本项目是 **NUS EEC4400 课程（Sem 1 AY25/26）** 的课程设计作业。
目标是通过实现四种深度强化学习（Deep Reinforcement Learning, DRL）算法，比较它们在经典控制任务 **Cart-Pole** 环境中的性能与稳定性。

四种实现的算法为：

1. **Q-Network**（基于神经网络的 Q-Learning，无经验回放和目标网络）
2. **Naïve DQN**（单网络版本）
3. **DQN**（含目标网络的标准 [DQN](https://arxiv.org/abs/1312.5602)）
4. **Double DQN (DDQN)**（双网络结构，减轻过估计偏差的 [DDQN](https://arxiv.org/abs/1509.06461)）

学生需独立实现并评估其中一种算法，并与组员共同比较结果。


## 🧩 环境与依赖

本项目基于 **Python + Jupyter Notebook** 实现。

### 必要依赖

```bash
pip install gymnasium torch tensorboard numpy matplotlib
```

### 推荐环境

* Python ≥ 3.9
* Jupyter Notebook
* TensorBoard
* 无需使用高级强化学习库（如 Stable Baselines3、RLlib 等）

---

## 🧠 实验内容与结构

本仓库基于课程提供的 **skeleton notebook** 扩展实现，包含以下部分：

| 模块                   | 内容                                               | 说明              |
| -------------------- | ------------------------------------------------ | --------------- |
| `1. 环境准备`            | Cart-Pole 环境初始化与基础操作                             | 提供完整示例代码        |
| `2. TensorBoard 配置`  | 日志目录与回调函数设置                                      | 提供完整示例代码        |
| `3. 算法实现`            | 实现指定的 DRL 算法（Q-Network / Naïve DQN / DQN / DDQN） | 需学生完成核心训练与评估逻辑  |
| `4. 性能评估`            | 绘制训练奖励、评估均值与方差                                   | 使用滑动平均窗口大小 = 20 |
| `5. 超参数调优`           | NN 与 RL 超参数分组设计与比较（hp-set1 vs hp-set2）           | 组内统一参数进行公平比较    |
| `6. 跨算法对比`           | 在同一图中绘制四种算法的表现                                   | 比较性能、稳定性与计算开销   |
| `7. TensorBoard 可视化` | 分析训练曲线与神经网络内部权重变化                                | 提供日志查看入口        |
| `8. 额外探索`            | 个性化实验与改进                                         | 可选内容            |

---

## ⚙️ 运行说明

1. 克隆仓库：

   ```bash
   git clone <your_repo_url>
   cd EEC4400-CartPole
   ```

2. 启动 Jupyter：

   ```bash
   jupyter notebook
   ```

3. 打开 `EEC4400-GroupXX.ipynb`
   按顺序运行所有代码单元。

4. 查看训练日志：

   ```bash
   tensorboard --logdir eec4400_logs
   ```

   打开浏览器访问 `http://localhost:6006`。

---

## 🔍 实验结果与分析

每种算法需分别生成两组实验结果：

* **Baseline Policy**：使用初始超参数组（`NN-hp-set1`, `RL-hp-set1`）训练。
* **Alternative Policy**：调整后的超参数组（`NN-hp-set2`, `RL-hp-set2`）训练。

实验中需：

* 比较不同超参数下的性能变化；
* 绘制训练奖励与评估奖励曲线；
* 分析神经网络内部参数与权重；
* 对比不同算法的稳定性与训练时间。

---

## 📊 报告要求（Report）

最终报告应包括以下部分：

1. **神经网络结构与超参数设计理由**
2. **实验结果与性能分析（含 TensorBoard 可视化）**
3. **四种算法的对比分析**

   * 任务完成效果
   * 收敛速度
   * 稳定性
   * 计算成本
4. **额外探索或改进（可选）**
5. **参考文献与贡献声明**

报告需控制在 **10 页以内**，以图表和实验数据支撑分析。

---

## 👥 分工与职责

| 学生编号      | 负责算法                            | 主要任务            |
| --------- | ------------------------------- | --------------- |
| Student 1 | Q-Network                       | 代码实现、滑动平均绘图函数   |
| Student 2 | Naïve DQN                       | 单网络版本实现与训练分析    |
| Student 3 | DQN                             | 含目标网络的标准 DQN 实现 |
| Student 4 | DDQN                            | 双网络结构实现与性能比较    |
| 所有成员      | 参数设计、跨算法对比、报告撰写、TensorBoard 可视化 |                 |

---

## 🗂️ 文件结构示例

```
EEC4400-CartPole/
│
├── EEC4400-GroupXX.ipynb         # 主实验文件
├── eec4400_logs/                 # TensorBoard 日志
├── README.md                     # 项目说明文件（本文件）
└── report/
    └── EEC4400-GroupXX-Report.pdf
```

---

## 🧾 引用与参考

* Mnih et al., “Playing Atari with Deep Reinforcement Learning,” *arXiv:1312.5602*
* Hasselt et al., “Deep Reinforcement Learning with Double Q-Learning,” *arXiv:1509.06461*
* OpenAI Gymnasium 文档：[https://gymnasium.farama.org/](https://gymnasium.farama.org/)