
# Q&A
## 降级 pip 与 setuptools
```shell
# 查看版本
pip --version
pip show setuptools
# 安装指定版本
python -m pip install pip==21.0
pip install setuptools==65.5.0
```


# 代码分析
main为游戏主题


# 库介绍
## stable-baselines3
主要用于强化学习（Reinforcement Learning, RL）任务。
1. Monitor
用途: Monitor 用于监控和记录环境的表现，比如每个回合的奖励、步数等信息。这对于分析模型的学习过程和调试非常有用。<br>
使用原理: Monitor 通过包装环境，使得每次调用 step() 和 reset() 方法时，都记录下奖励和状态信息，并将其保存到日志文件中。<br>

2. SubprocVecEnv
用途: SubprocVecEnv 用于并行化多个环境的运行，以加快训练速度。强化学习通常需要大量的样本，而并行运行多个环境能够更高效地收集这些样本。<br>
使用原理: SubprocVecEnv 通过创建多个子进程来运行环境，每个进程独立收集样本，然后将样本合并返回。这样可以充分利用多核 CPU 的性能。<br>

3. CheckpointCallback
用途: CheckpointCallback 用于在训练过程中保存模型的检查点，以便在训练失败或中断时能够恢复。<br>
使用原理: 在每次训练的指定周期内，CheckpointCallback 会将当前的模型权重保存到文件中。<br>

## Gym
Gym 是一个用于开发和比较强化学习算法的工具包，它提供了一系列标准化的环境。由 OpenAI 开发，Gym 旨在简化强化学习的实验过程，使研究人员和开发者能够轻松创建、测试和共享他们的算法。<br>

### Gym 环境的特点和用途
#### 标准化的接口：
Gym 提供了统一的接口，使得不同的环境可以使用相同的方法进行交互。主要的方法包括 reset()、step(action) 和 render()。

#### 多样化的环境：
Gym 提供了多种类型的环境，包括：
* 经典控制问题（如 CartPole、MountainCar）
* Atari (雅达利) 游戏（如 Pong、Breakout）
* 机器人仿真（如 MuJoCo、Robotics）
* 自定义环境：用户也可以根据需求创建新的环境。

#### 易于使用：
Gym 的简单接口使得即使是初学者也能快速上手，进行实验和调试。
#### 用于算法开发和测试：
Gym 环境允许研究人员和开发者在相同的环境中测试不同的算法，从而进行公平比较。
#### 支持多种强化学习库：
Gym 可以与多种强化学习库配合使用，如 Stable Baselines、RLlib、TensorFlow Agents 等。