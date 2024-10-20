import gym
import pygame
import numpy as np
import time

# CartPole-v1 部署成磁盘输入

# 初始化 pygame
pygame.init()

# 创建 Gym 环境
env = gym.make('CartPole-v1')

# 设置窗口
screen = pygame.display.set_mode((600, 400))
pygame.display.set_caption("CartPole Control")

# 重置环境
observation = env.reset()

# 设置时钟
clock = pygame.time.Clock()

# 游戏循环
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 获取键盘输入
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        action = 0  # 向左推
    elif keys[pygame.K_RIGHT]:
        action = 1  # 向右推
    else:
        action = None  # 不动作

    done = None
    if action != None:
        print("action: %d" % (action))
        # 采取动作
        observation, reward, done, info = env.step(action)

        # # 渲染环境
        # frame = env.render()
        # frame = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        # screen.blit(pygame.transform.scale(frame, (600, 400)), (0, 0))
        # pygame.display.flip()

    if done:
        print("game is done to reset")
        observation = env.reset()

    # 渲染环境
    env.render()

    # 控制帧率
    clock.tick(60)
    time.sleep(0.1)

# 结束
env.close()
pygame.quit()
