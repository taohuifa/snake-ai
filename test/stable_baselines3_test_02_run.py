import gym
import os
import pygame
import time
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3_test_02 import *

if __name__ == '__main__':
    print("start pid: %d" % (os.getpid()))

    env = Monitor(gym.make(game_name))
    # 加载模型
    model = PPO.load(f'./logs/{model_file}')

    # 初始化 pygame
    pygame.init()
    # 设置窗口
    screen = pygame.display.set_mode((600, 400))
    pygame.display.set_caption("CartPole Control")
    clock = pygame.time.Clock()
    # 运行环境
    obs = env.reset()
    for epoch in range(1000):  # 运行1000个时间步
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        print(f"epoch: {epoch} action:{action} rewards:{rewards} info:{info}")
        env.render()  # 渲染环境

        if dones:
            print(f"game is finish: {info}")
            time.sleep(3)
            break

        # 控制帧率
        clock.tick(60)

    # 关闭环境
    env.close()
