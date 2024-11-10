import gym
import os
import pygame
import time
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3_test_02 import *
import common

if __name__ == '__main__':
    print("start pid: %d" % (os.getpid()))
    reset_times = 10
    finish_times = 0

    env = Monitor(common.gym_make(game_name))
    # 加载模型
    model = PPO.load(f'./logs/{model_file}')

    # 初始化 pygame
    pygame.init()
    # 设置窗口
    # screen = pygame.display.set_mode((600, 600))
    # pygame.display.set_caption("CartPole Control")
    clock = pygame.time.Clock()
    # 运行环境
    obs = env.reset()
    for epoch in range(1000):  # 运行1000个时间步
        action, _states = model.predict(obs)
        print(f"action: {action}")
        obs, rewards, dones, info = env.step(action)
        print(f"epoch: {epoch} action:{action} rewards:{rewards} info:{info}")
        env.render()  # 渲染环境

        if dones:
            print(f"game is finish: {info} {finish_times}/{reset_times}")
            finish_times += 1
            time.sleep(1)
            obs = env.reset()
            time.sleep(1)
            if finish_times >= reset_times:
                break

        # 控制帧率
        clock.tick(3)

    # 关闭环境
    env.close()