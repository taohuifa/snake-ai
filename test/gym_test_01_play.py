import gym
import pygame
import numpy as np
import time
import os
import common
import matplotlib.pyplot as plt


# 遍历并输出 gym 环境注册信息
for env in gym.envs.registry.all():
    print(env)

# gameName = 'CartPole-v1'
gameName = 'MountainCar-v0'
# gameName = 'CubeCrash-v0'
# gameName = 'Pong-v0'
tick = 10

# 创建 Gym 环境
if gameName in {'CartPole-v1', 'CartPole-v0'}:
    env = gym.make(gameName)  # CartPole-v1（平衡车）
    action_map = {
        pygame.K_LEFT: 0,
        pygame.K_RIGHT: 1,
        0: None,  # 默认
    }
elif gameName == 'MountainCar-v0':
    env, tick = common.gym_make(gameName)  # MountainCar-v0（山地车）
    action_map = {
        pygame.K_LEFT: 0,  # 向左推
        pygame.K_RIGHT: 2,  # 向右推
        0: 1,  # 默认
    }
elif gameName == 'CubeCrash-v0':
    env = gym.make(gameName)
    action_map = {
        pygame.K_LEFT: 1,  # 向左推
        pygame.K_RIGHT: 2,  # 向右推
        0: 0,  # 默认
    }
elif gameName == 'Pong-v0':
    env = gym.make(gameName)
    action_map = {
        pygame.K_LEFT: 0,  # 向左推
        pygame.K_RIGHT: 2,  # 向右推
        0: 1,  # 默认
    }
else:
    print(f"unknow game {gameName}")
    os._exit(1)

# 初始化 pygame
pygame.init()

# 设置窗口
# screen = pygame.display.set_mode((800, 600))  # 调整窗口大小
screen = pygame.display.set_mode((400, 300))  # 调整窗口大小
pygame.display.set_caption("Pong Control")

# 重置环境
observation = env.reset()

print(f"动作空间的形状: {env.action_space.shape}, 可能的动作数量: {env.action_space.n}")
for action in range(env.action_space.n):
    print(f"  动作 {action}: {env.action_space.contains(action)}")

# 设置时钟
clock = pygame.time.Clock()

# 在主游戏循环前添加reward收集列表
running = True
idx = 0
rewards = []  # 新增：用于收集reward
pos = []
speeds = []
episode_rewards = []  # 新增：用于收集每个episode的总reward

while running:
    idx = idx + 1

    # 渲染环境
    frame = env.render(mode='rgb_array')  # 获取渲染的帧
    frame = np.transpose(frame, (1, 0, 2))  # 转置以适应 Pygame 的格式
    frame_surface = pygame.surfarray.make_surface(frame)  # 创建 Pygame 表面

    # 自动适应窗口尺寸
    frame_surface = pygame.transform.scale(frame_surface, screen.get_size())  # 缩放到窗口大小
    screen.blit(frame_surface, (0, 0))  # 将表面绘制到屏幕上
    pygame.display.flip()  # 更新显示

    # 控制帧率
    clock.tick(tick)
    # time.sleep(0.1)

    # 检查退出
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 获取键盘输入
    keys = pygame.key.get_pressed()
    action = next((action for key, action in action_map.items() if keys[key]), action_map[0])  # 默认动作为 1
    # print(f"idx: {idx} action: {action}")

    done = None
    if action is None:
        continue

    # 采取动作
    observation, reward, done, info = env.step(action)
    # action_mask = env.get_action_mask()

    # if idx % 30 == 0:
    # print(f"idx: {idx} obs: {observation.reshape(-1)} action: {action} reward: {reward} info: {info}")
    # print(f"idx: {idx} obs: {observation.reshape(-1)} action: {action} reward: {reward} action_mask: {action_mask}")

    # 在step之后收集reward
    obs = observation.reshape(-1)
    rewards.append(reward)  # 新增：收集reward
    pos.append(obs[0])
    speeds.append(obs[1] * 10)

    if done:
        print("game is done to reset")
        total_reward = sum(rewards)  # 计算本轮总reward
        episode_rewards.append(total_reward)  # 保存本轮总reward
        print(f"Episode finished with total reward: {total_reward} {reward}")

        # 绘制reward时序图
        plt.figure(figsize=(10, 5))
        plt.plot(rewards, label='Rewards', color='blue')
        plt.plot(pos, label='Position', color='red')
        plt.plot(speeds, label='Speed', color='green')
        plt.title(f'Rewards over time (Episode total: {total_reward:.2f})')
        plt.xlabel('Steps')
        plt.ylabel('Values')
        plt.grid(True)
        plt.legend()  # 添加图例
        plt.show()

        # 重置reward收集列表
        rewards = []
        pos = []
        speeds = []

        observation = env.reset()
        time.sleep(1)
        continue

# 在游戏结束时绘制所有episode的总reward
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards, label='Episode Rewards', color='purple')
plt.title('Total Rewards per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True)
plt.legend()  # 添加图例
plt.show()

# 结束
env.close()
pygame.quit()
