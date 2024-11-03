import gym  # 导入OpenAI Gym库，用于创建和管理环境
import numpy as np  # 导入NumPy库，用于数值计算
import torch  # 导入PyTorch库，用于构建和训练神经网络
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.optim as optim  # 导入PyTorch的优化器模块
from torch.distributions import Categorical  # 导入分类分布，用于处理离散动作

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

# 实现PPO算法


class PPO:
    def __init__(self, env, learning_rate=3e-4, gamma=0.99, epsilon=0.2, epochs=10):
        self.env = env  # 环境
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # PPO的剪切参数
        self.epochs = epochs  # 更新的轮数

        print("env.observation_space.shape: ", env.observation_space.shape)
        # 初始化策略网络
        self.policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
        # 初始化价值网络
        self.value = ValueNetwork(env.observation_space.shape[0])
        # 使用Adam优化器
        self.optimizer = optim.Adam(list(self.policy.parameters()) + list(self.value.parameters()), lr=learning_rate)

    def get_action(self, state):
        state = torch.FloatTensor(state)  # 将状态转换为张量
        logits = self.policy(state)  # 通过策略网络获取动作的logits
        dist = Categorical(logits=logits)  # 创建分类分布
        action = dist.sample()  # 从分布中采样动作
        return action.item(), dist.log_prob(action)  # 返回动作及其对数概率

    def compute_gae(self, rewards, values, next_value, dones, gamma=0.99, lam=0.95):
        advantages = []  # 存储优势值
        last_advantage = 0  # 上一个优势值
        last_value = next_value  # 下一个状态的价值

        # 反向计算优势值
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * last_value * (1 - dones[t]) - values[t]  # 计算TD误差
            last_advantage = delta + gamma * lam * (1 - dones[t]) * last_advantage  # 计算优势值
            advantages.insert(0, last_advantage)  # 将优势值插入到列表的开头
            last_value = values[t]  # 更新最后的价值

        returns = np.array(advantages) + values  # 计算回报
        advantages = np.array(advantages)  # 转换为NumPy数组
        return returns, advantages  # 返回回报和优势值

    def learn(self, total_timesteps):
        for timestep in range(total_timesteps):
            state = self.env.reset()  # 重置环境
            done = False  # 任务是否完成
            total_reward = 0  # 总奖励

            # 存储状态、动作、奖励、对数概率、价值和完成标��
            states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []

            while not done:
                action, log_prob = self.get_action(state)  # 获取动作和对数概率
                next_state, reward, done, _ = self.env.step(action)  # 执行动作并获取下一个状态和奖励
                value = self.value(torch.FloatTensor(state)).item()  # 计算当前状态的价值
                # print(f"step: {len(states)} reward: {reward} done: {done}")
                # 存储数据
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value)
                dones.append(done)

                state = next_state  # 更新状态
                total_reward += reward  # 累加奖励

            next_value = self.value(torch.FloatTensor(next_state)).item()  # 计算下一个状态的价值
            returns, advantages = self.compute_gae(rewards, values, next_value, dones)  # 计算回报和优势值

            # 转换为张量
            states = torch.FloatTensor(np.array(states)).detach()
            actions = torch.LongTensor(actions).detach()
            returns = torch.FloatTensor(returns).detach()
            advantages = torch.FloatTensor(advantages).detach()
            old_log_probs = torch.stack(log_probs).detach()  # 将对数概率堆叠成张量

            for _ in range(self.epochs):
                new_log_probs = Categorical(logits=self.policy(states)).log_prob(actions)  # 计算新的对数概率
                new_values = self.value(states).detach().squeeze()  # 计算新的价值并分离计算图

                # 计算比率和损失
                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()  # 计算策略损失
                critic_loss = nn.MSELoss()(new_values, returns)  # 计算价值损失

                loss = actor_loss + 0.5 * critic_loss  # 总损失

                self.optimizer.zero_grad()  # 清空梯度
                loss.backward()  # 反向传播
                self.optimizer.step()  # 更新参数

            if timestep % 100 == 0:
                print(f"timestep: {timestep} Total reward: {total_reward} shape: {returns.shape[0]}")  # 打印总奖励


if __name__ == '__main__':
    # env = gym.make('CartPole-v1')  # 创建CartPole环境
    env = gym.make('MountainCar-v0')  
    ppo = PPO(env, epochs=10)  # 初始化PPO算法
    ppo.learn(total_timesteps=1)  # 开始学习
    env.close()  # 关闭环境
