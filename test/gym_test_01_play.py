import gym
import pygame
import numpy as np
import time
import os


# 遍历并输出 gym 环境注册信息
for env in gym.envs.registry.all():
    print(env)

# gameName = 'CartPole-v1'
gameName = 'MountainCar-v0'
# gameName = 'CubeCrash-v0'
# gameName = 'Pong-v0'

# 创建 Gym 环境
if gameName in ['CartPole-v1', 'CartPole-v0']:
    env = gym.make(gameName)  # CartPole-v1（平衡车）
    action_map = {
        pygame.K_LEFT: 0,
        pygame.K_RIGHT: 1,
        0: None,  # 默认
    }
elif gameName == 'MountainCar-v0':
    env = gym.make(gameName)  # MountainCar-v0（山地车）
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

# 游戏循环
running = True
idx = 0
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
    clock.tick(10)
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
    print(f"idx: {idx} action: {action} reward: {reward} info: {info}")

    if done:
        print("game is done to reset")
        observation = env.reset()
        time.sleep(1)
        # break
        continue


# 结束
env.close()
pygame.quit()
