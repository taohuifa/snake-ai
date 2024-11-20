import gym  # 导入OpenAI Gym库，用于创建和管理环境
import numpy as np  # 导入NumPy库，用于数值计算
import torch  # 导入PyTorch库，用于构建和训练神经网络
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.optim as optim  # 导入PyTorch的优化器模块
from torch.distributions import Categorical  # 导入分类分布，用于处理离散动作
import zipfile  # 导入zipfile库，用于处理zip文件
import os  # 导入os库，用于文件操作
import common
from stable_baselines3.common.monitor import Monitor
import math
import datetime, time


# 创建logger实例
logger = common.setup_logger('ppo_training',
                             f'logs/training_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

# 定义策略网络


class CnnPolicy(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(CnnPolicy, self).__init__()
        # 定义一个包含卷积层的神经网络
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),  # 输入通道数为input_channels，输出通道数为32
            nn.ReLU(),  # 激活函数
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 输入通道数为32，输出通道数为64
            nn.ReLU(),  # 激活函数
            nn.Flatten(),  # 展平层
            nn.Linear(64 * 8 * 8, 128),  # 假设输入图像大小为64x64，展平后连接到全连接层
            nn.ReLU(),  # 激活函数
            nn.Linear(128, num_actions)  # 输出层，输出动作的logits
        )

    def forward(self, x):
        return self.conv(x)  # 前向传播


# 定义策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        # 定义一个包含两个隐藏层的全连接神经网络
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),  # 输入层到第一个隐藏层
            nn.ReLU(),  # 激活函数
            nn.Linear(64, 64),  # 第一个隐藏层到第二个隐藏层
            nn.ReLU(),  # 激活函数
            nn.Linear(64, output_dim)  # 第二个隐藏层到输出层
        )

    def forward(self, x):
        return self.fc(x)  # 前向传播


# 定义价值网络
class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        # 定义一个包含两个隐藏层的全连接神经网络
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),  # 输入层到第一个隐藏层
            nn.ReLU(),  # 激活函数
            nn.Linear(64, 64),  # 第一个隐藏层到第二个隐藏层
            nn.ReLU(),  # 激活函数
            nn.Linear(64, 1)  # 第二个隐藏层到输出层（输出一个值）
        )

    def forward(self, x):
        return self.fc(x)  # 前向传播


def create_policy(policy_name: str, input_dim: int, output_dim: int):
    if policy_name == "CnnPolicy":
        return CnnPolicy(input_dim, output_dim)
    return PolicyNetwork(input_dim, output_dim)


# 实现PPO算法
class PPO:
    def __init__(self, env, policy_name="", device="cuda", learning_rate=3e-4, gamma=0.99,
                 epsilon=0.2, epochs=100):
        self.env = env  # 环境
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # PPO的剪切参数
        self.epochs = epochs  # 更新的轮数

        input_dim = np.prod(env.observation_space.shape)
        logger.info(f"env.observation_space: {env.observation_space.shape}, "
                    f"input_dim: {input_dim}, "
                    f"env.action_space.n: {env.action_space.n}")

        # 设置设备
        self.device = device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device)

        # 初始化策略网络
        self.policy = create_policy(policy_name, input_dim, env.action_space.n).to(self.device)
        # 初始化价值网络
        self.value = ValueNetwork(input_dim).to(self.device)
        # 使用Adam优化器
        params = list(self.policy.parameters()) + list(self.value.parameters())
        # self.optimizer = optim.Adam(params, lr=learning_rate)
        self.optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9)

    def predict(self, state):
        return self.get_action(state)

    def get_action(self, state, use_action_mask: bool = True):
        logits = self.policy(state)  # 通过策略网络获取动作的logits

        if use_action_mask and self.env.get_action_mask is not None:
            action_masks = torch.FloatTensor(self.env.get_action_mask()).to(self.device)  # 获取动作掩码
            logits = logits + (1 - action_masks) * -1e10  # 将被抹掉的动作的logits设置为一个很小的值

        dist = Categorical(logits=logits)  # 创建分类分布
        action = dist.sample()  # 从分布中采样动作
        log_prob = dist.log_prob(action)
        return action.item(), log_prob  # 返回动作及其对数概率

    # 计算优势和回报率
    # gamma 用于折扣未来奖励的现值
    # lam 则用于控制优势的平滑程度

    def compute_gae(self, rewards, values, next_value, dones, gamma=0.99, lam=0.95):
        advantages = []  # 存储优势值
        last_advantage = 0  # 上一个优势值
        last_value = next_value  # 下一个状态的价值

        # 反向计算优势值
        for t in reversed(range(len(rewards))):
            # r = 1.0 - (len(rewards) - t) / len(rewards)
            r = (len(rewards) - t) / len(rewards)
            # 损失误差: 奖励值 + (gamma * 下一次的价值) * (如果成功, 则下一次的DT这次的损失为0)
            delta = rewards[t] + gamma * last_value * (1 - dones[t]) - values[t]  # 计算TD误差
            last_advantage = delta + gamma * lam * (1 - dones[t]) * last_advantage * r  # 计算优势值
            # print(f"compute_gae[{t}]: reward: {rewards[t]} value: {values[t]} -> {last_value}, done: {dones[t]}. delta: {delta}, last_advantage: {last_advantage} r: {r}")
            advantages.insert(0, last_advantage)  # 将优势值插入到列表的开头
            last_value = values[t]  # 更新最后的价值

        returns = np.array(advantages) + values  # 计算回报
        advantages = np.array(advantages)  # 转换为NumPy数组
        return returns, advantages  # 返回回报和优势值

    def learn(self, total_timesteps, use_action_mask: bool = True):
        idx = 0
        for timestep in range(total_timesteps):
            state = self.env.reset()  # 重置环境, state: ([2,])
            done = False  # 任务是否完成
            total_reward = 0  # 总奖励

            # 存储状态、动作、奖励、对数概率、价值和完成标
            states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []

            # 遍历尝试操作游戏, 第一次通常是200次就失败
            while not done:
                obs = torch.FloatTensor(state).detach().to(self.device)
                # print(f"obs: {state.shape} {obs.shape}")

                # 计算策略和价值
                action, log_prob = self.get_action(obs, use_action_mask)  # 获取动作和对数概率
                value = self.value(obs).item()  # 计算当前状态的价值
                # 计算步骤
                next_state, reward, done, _ = self.env.step(action)  # 执行动作并获取下一个状态和奖励

                # 打印数据
                idx += 1
                if idx % 10000 == 0:
                    logger.info(f"[{timestep}/{idx}] step: {len(states)} "
                                f"action: {action} reward: {reward} done: {done} "
                                f"state: {next_state.reshape(-1)}")

                # 存储数据(把每个步骤的处理结果和回包都记录起来)
                states.append(np.array(obs.cpu().view(-1)))  # 确保每个状态都是一维的张量
                actions.append(action)  # size: (1)
                rewards.append(reward)  # size: (1)
                log_probs.append(log_prob)  # size: (1)
                values.append(value)  # size: (1)
                dones.append(done)  # size: (1)

                state = next_state  # 更新状态
                total_reward += reward  # 累加奖励

            # 通过使用最后一个状态的价值，算法能够更准确地评估当前策略的表现，并进行相应的调整。
            next_state = torch.FloatTensor(next_state) .to(self.device)
            next_value = self.value(next_state).item()  # 计算下一个状态的价值
            # print(f"next_state: {next_state.shape} -> {next_state}")
            # 这里通过拿到1个最终游戏结果值, 倒序计算之前每一步的优势
            returns, advantages = self.compute_gae(rewards, values, next_value, dones)  # 计算回报和优势值
            # print(f"timestep: {timestep} returns: {len(returns)} {len(advantages)} from {len(values)} -> {next_value}: {len(rewards)}")

            # 转换为张量, 第一次通常执行200次就失败, N=200
            # print(f"{states}")
            states = torch.FloatTensor(np.array(states)).detach().to(self.device)  # shape: ([N,2])
            actions = torch.LongTensor(actions).detach().to(self.device)  # shape: ([N]), 每次一个动作
            old_log_probs = torch.stack(log_probs).detach().to(self.device)  # 将对概率, 每一次选择这个动作的可能性高低, shape: ([N])
            returns = torch.FloatTensor(returns).detach().to(self.device)  # shape: ([N]), 每次一个回报值(起一次操作变化)
            advantages = torch.FloatTensor(advantages).detach().to(self.device)  # 优势情况, shape: ([N])
            self.optimizer.zero_grad()  # 清空梯度

            # print(f"timestep: {timestep} Total reward: {total_reward}, shape: {states.shape} {returns.shape}, old_log_probs: {old_log_probs.shape}")
            if (timestep % 100 == 0) or (timestep == total_timesteps - 1):
                logger.info(f"timestep: {timestep} Total reward: {total_reward} "
                            f"step: {returns.shape[0]}")

            for epoch in range(self.epochs):
                # logits = torch.tensor([0.5, 1.0, 0.1])  # 三个动作的 logits
                # Categorical(logits=self.policy(states)) # 采样一个动作
                # action = dist.sample() # 采样一个动作
                # log_prob = dist.log_prob(action) # 计算所选动作的对数概率

                # 概述
                # 1. 用决策模型, 重新计算当下状态下, 每个动作的概率(logit)
                # 2. 用分类Categorical, 计算得出

                # 用决策模型对每个状态重新进行决策(因为每一轮决策模型会更新)
                logits = self.policy(states)  # size: (N,A), 每个场景对应每个动作的概率
                new_log_probs = Categorical(logits=logits).log_prob(actions)  # 计算新的对数概率, 计算下新权重对每个状态对应的动作的新概率, size: ([200])

                # 重新计算下每个状态下的价值(因为因为每一轮价值模型会更新)
                new_values = self.value(states).detach().squeeze()  # 计算新的价值并分离计算图, 对每个状态下的价值重新计算, torch.Size([200])
                # print(f"epoch: {epoch} -> {new_log_probs.shape}, {new_values.shape}")

                # 计算新旧对数概率的比率
                # -> 这个比率用于衡量策略更新的幅度，确保在PPO算法中，策略不会过度更新。
                ratio = (new_log_probs - old_log_probs).exp()  # 每一步骤的概率值, size (N)
                surr1 = ratio * advantages  # 差值 * 当前步骤时的优势 size: (N)
                # 损失值裁剪: (1 - self.epsilon, 1 + self.epsilon) 默认: (0.8 ~ 1.2)
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages  # 计算过裁剪后的插值 size: (N)
                # 计算策略损失(取最小值)
                actor_loss = -torch.min(surr1, surr2).mean()  # 取最小的策略损失值, size: 标量

                critic_loss = nn.MSELoss()(new_values, returns)     # 计算价值损失(拿新的价值, 跟之前的回包做比对), size: 标量

                l = actor_loss + 0.5 * critic_loss   # 总损失
                # print(f"epoch: {epoch}, loss: {l}, ratio: {ratio.shape} actor_loss: {actor_loss} critic_loss: {critic_loss} ")
                if timestep % 100 == 0 and (epoch == self.epochs):
                    logger.info(f"epoch: {epoch}, loss: {l}")

                self.optimizer.zero_grad()  # 清空梯度
                l.backward()  # 反向传播，保留计算图
                self.optimizer.step()  # 更新参数

    def save_model(self, file_name):
        # 保存模型的状态字典
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
        }, file_name)

    def load_model(self, file_name):
        # 加载模型的状态字典
        checkpoint = torch.load(file_name)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])

    def _check_action_validity(self, action):
        """
        检查给定的动作是否有效。

        参数:
        action: 要检查的动作

        返回:
        bool: 如果动作有效返回True，否则返回False
        """
        # 检查动作是否在有效范围内
        if action < 0 or action >= self.env.action_space.n:
            return False  # 动作超出范围，返回False

        if self.env.get_action_mask is not None:
            action_masks = self.env.get_action_mask()  # 获取动作掩码
            return action_masks[action]  # 检查该动作是否有效

        return True  # 如果没有动作掩码，默认所有动作有效


# game_name = 'CartPole-v1'
game_name = 'MountainCar-v0'
# game_name = 'game_gridworld'
model_file = f"{game_name.replace('-','_').lower()}_test03"


if __name__ == '__main__':
    start_time = time.time()  # 记录开始时间

    env, _ = common.gym_make(game_name)
    env = Monitor(env)
    # env = gym.make('MountainCar-v0')
    ppo = PPO(env, epochs=30, device="cuda")  # 初始化PPO算法, cuda, cpu
    logger.info(f"Starting training for {game_name}")
    ppo.learn(total_timesteps=100)  # 开始学习

    # 保存模型
    save_file = f"logs/{model_file}.zip"
    ppo.save_model(save_file)

    # 计算并记录总运行时间
    end_time = time.time()
    duration = end_time - start_time
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)

    logger.info(f"Training finished, model saved to {save_file}")
    logger.info(f"Total running time: {hours:02d}:{minutes:02d}:{seconds:02d}")

    env.close()
