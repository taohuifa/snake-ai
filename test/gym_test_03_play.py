import gym
import pygame
import numpy as np
import time
import os
import common
from pynput import keyboard


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
        0: 1,  # 默认
        keyboard.Key.left: 1,
        keyboard.Key.right: 2,
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

# 重置环境
observation = env.reset()

common.init_keyboard_listener(action_map)
sleep_time = 1 / tick
print(f"动作空间的形状: {env.action_space.shape}, 可能的动作数量: {env.action_space.n}, sleep: {sleep_time}")
for action in range(env.action_space.n):
    print(f"  动作 {action}: {env.action_space.contains(action)}")

# 游戏循环
running = True
idx = 0
while running:
    idx = idx + 1

    # 渲染环境
    frame = env.render(mode='rgb_array')  # 获取渲染的帧

    # 控制帧率
    time.sleep(sleep_time)

    # 获取键盘输入
    # keys = pygame.key.get_pressed()
    # action = next((action for key, action in action_map.items() if keys[key]), action_map[0])  # 默认动作为 1
    action = common.current_action
    # print(f"idx: {idx} action: {action}")

    done = None
    if action is None:
        continue

    # 采取动作
    observation, reward, done, info = env.step(action)
    
    # if idx % 30 == 0:
    # print(f"idx: {idx} obs: {observation.reshape(-1)} action: {action} reward: {reward} info: {info}")
    # print(f"idx: {idx} obs: {observation.reshape(-1)} action: {action} reward: {reward} action_mask: {action_mask}")

    if done:
        print("game is done to reset")
        observation = env.reset()
        time.sleep(1)
        # break
        continue


# 结束
env.close()
pygame.quit()
