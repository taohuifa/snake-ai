import sys
import os
import gym  # 导入OpenAI Gym库，用于创建和管理环境
import pygame
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from stable_baselines3_test_03 import *


if __name__ == '__main__':
    env = gym.make(game_name)
    
    # 加载模型
    load_file = f"logs/{model_file}.zip"
    model = PPO(env, epochs=20)
    model.load_model(load_file)

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
