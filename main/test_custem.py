
import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(cur_dir))
sys.path.append(os.path.dirname(cur_dir) + "/test")
print(f"cur_dir: {cur_dir}")

import common
from stable_baselines3_test_03 import *
from stable_baselines3.common.monitor import Monitor
import math
import datetime
import time
import pygame
from snake_game_custom_wrapper_cnn import *
from train_custem import *


if __name__ == '__main__':
    env = SnakeEnv(silent_mode=False)
    tick = 10

    # 加载模型
    load_file = f"logs/{model_file}.zip"
    model = PPO(env, epochs=20)
    model.load_model(load_file)

    # 初始化 pygame
    pygame.init()
    # 设置窗口
    # screen = pygame.display.set_mode((600, 600))
    # pygame.display.set_caption("CartPole Control")

    clock = pygame.time.Clock()
    # 运行环境
    obs = env.reset()
    for epoch in range(10000):  # 运行1000个时间步
        # 控制帧率
        clock.tick(tick)

        # 预测执行
        obs = torch.FloatTensor(obs).reshape(-1)
        # action, _states = model.predict(obs)
        action, _ = model.get_action(obs, True)
        obs, rewards, dones, info = env.step(action)
        print(f"epoch: {epoch} action:{action} rewards:{rewards} obs: {obs.reshape(-1)}")
        # # 渲染环境
        # frame = env.render(mode='rgb_array')  # 获取渲染的帧
        # # frame = np.transpose(frame, (1, 0, 2))  # 转置以适应 Pygame 的格式
        # frame_surface = pygame.surfarray.make_surface(frame)  # 创建 Pygame 表面

        # # 自动适应窗口尺寸
        # frame_surface = pygame.transform.scale(frame_surface, screen.get_size())  # 缩放到窗口大小
        # screen.blit(frame_surface, (0, 0))  # 将表面绘制到屏幕上
        # # pygame.display.flip()  # 更新显示
        env.render()

        if dones:
            print(f"game is finish: {info}")
            time.sleep(1)
            obs = env.reset()
            env.render()
            time.sleep(1)
            continue

    # 关闭环境
    env.close()
