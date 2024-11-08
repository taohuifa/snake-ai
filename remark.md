
# Q&A
## 降级 pip 与 setuptools
```shell
conda activate  SnakeAI
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

#### 游戏详情介绍
```shell
* CartPole-v1（平衡车）：
描述：一个倒立摆安装在可以左右移动的小车上。
目标：通过左右移动小车来保持杆子直立。
状态空间：小车位置、速度，杆子角度、角速度。
动作空间：向左或向右推小车。
* MountainCar-v0（山地车）：
描述：一辆小车位于两座山之间的谷底。
目标：通过来回摆动积累动能，最终爬上右边的山顶。
状态空间：小车位置和速度。
动作空间：向左加速、不动、向右加速。
* Pendulum-v1（钟摆）：
描述：一个倒立摆，可以360度旋转。
目标：将摆杆保持在垂直向上的位置。
状态空间：摆杆角度和角速度。
动作空间：施加的扭矩（连续值）。
* Acrobot-v1（杂技机器人）：
描述：两个连接的杆，第一个杆固定在一点，第二个杆可以自由旋转。
目标：摆动下端使其达到一定高度。
状态空间：两个关节的角度和角速度。
动作空间：对第二个关节施加正扭矩、负扭矩或不施加扭矩。
* LunarLander-v2（月球着陆器）：
描述：模拟月球着陆器在月球表面着陆。
目标：安全、平稳地将着陆器降落在指定区域。
状态空间：位置、速度、角度、角速度等。
动作空间：主引擎和侧向推进器的控制。
* BipedalWalker-v3（双足行走）：
描述：一个具有两条腿的机器人。
目标：控制机器人平稳地向前行走。
状态空间：躯干和腿部的位置、角度、速度等。
动作空间：控制髋关节和膝关节的扭矩。
* Pong-v0（乒乓球）：
描述：经典的 Atari 乒乓球游戏。
目标：控制球拍击球，得分并防止对手得分。
状态空间：游戏画面（像素）。
动作空间：上移、下移或不动。
* FrozenLake-v1（冰湖）：
描述：在一个冰冻的湖面上行走，有些地方是冰洞。
目标：从起点安全地到达终点，避开冰洞。
状态空间：当前位置（离散）。
动作空间：上、下、左、右移动。
```

#### 易于使用：
Gym 的简单接口使得即使是初学者也能快速上手，进行实验和调试。
#### 用于算法开发和测试：
Gym 环境允许研究人员和开发者在相同的环境中测试不同的算法，从而进行公平比较。
#### 支持多种强化学习库：
Gym 可以与多种强化学习库配合使用，如 Stable Baselines、RLlib、TensorFlow Agents 等。

https://blog.csdn.net/lyx369639/article/details/127085462
https://www.cnblogs.com/tiandsp/p/18124932

https://www.cnblogs.com/cenariusxz/p/12666938.html

## sb3_contrib
### PPO（Proximal Policy Optimization，近端策略优化）是一种用于强化学习的算法
### MaskablePPO: 
MaskablePPO 是 Proximal Policy Optimization (PPO) 的一个变体，通过动作掩蔽机制来增强标准的PPO算法。动作掩蔽可以在动作空间中禁用某些动作，这对于某些有约束的环境非常有用，比如在某些状态下某些动作是非法的或者无效的。
```python
model = PPO('MlpPolicy', env, verbose=1)    // MlpPolicy 主要用于处理扁平的、连续的或离散的观察空间。它适合于那些输入数据是一个一维数组的情况
model = PPO(MultiInputPolicy, env, verbose=1) // MultiInputPolicy 适用于处理复杂的观察空间，特别是当观察空间是一个字典或包含多个输入的情况
```


### ActionMasker:
ActionMasker 是一个用于在环境中应用动作掩蔽的工具。它允许你在环境的某些状态下指定哪些动作是可用的，哪些不可用。



