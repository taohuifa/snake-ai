
import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(cur_dir))
sys.path.append(os.path.dirname(cur_dir) +"/test")
print(f"cur_dir: {cur_dir}")

import common
from stable_baselines3_test_03 import *
from stable_baselines3.common.monitor import Monitor
import math
import datetime
import time
from snake_game_custom_wrapper_cnn import *


# 创建logger实例
logger = common.setup_logger('ppo_training',
                             f'logs/training_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')


# game_name = 'CartPole-v1'
# game_name = 'MountainCar-v0'
game_name = 'snake'
model_file = f"{game_name.replace('-','_').lower()}_test03"


if __name__ == '__main__':
    start_time = time.time()  # 记录开始时间

    env = SnakeEnv(silent_mode=False)
    env = Monitor(env)
    # env = gym.make('MountainCar-v0')
    ppo = PPO(env, epochs=5, device="cuda")  # 初始化PPO算法, cuda, cpu
    logger.info(f"Starting training for {game_name}")
    ppo.learn(total_timesteps=1000)  # 开始学习

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
