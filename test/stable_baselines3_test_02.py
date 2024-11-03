import gym
import os
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback


# game_name = 'CartPole-v1'
game_name = 'MountainCar-v0'
model_file = f"{game_name.replace('-','_').lower()}_test02"

if __name__ == '__main__':
    print("start pid: %d" % (os.getpid()))

    # 创建 PPO 模型
    env = Monitor(gym.make(game_name))
    model = PPO('MlpPolicy', env, verbose=1)

    # 定义检查点回调
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/',
                                             name_prefix=model_file)

    # 训练模型
    model.learn(total_timesteps=10000, callback=checkpoint_callback)
    print(f"model learn finish: {model} -> {model_file}")

    # 评估模型
    model.save(f'./logs/{model_file}')  # 保存模型
    del model  # 删除模型以便重新加载

    # 关闭环境
    env.close()
