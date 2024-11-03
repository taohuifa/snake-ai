import gym
import os
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback


if __name__ == '__main__':
    print("start pid: %d" % (os.getpid()))
 

    # name_prefix = "ppo_cartpole"
    # game_name = 'CartPole-v1'
    name_prefix = "ppo_mountaincar"
    game_name = 'MountainCar-v0'
    env = Monitor(gym.make(game_name))

    # 创建 PPO 模型
    model = PPO('MlpPolicy', env, verbose=1)

    # 定义检查点回调
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/',
                                             name_prefix=name_prefix)

    # 训练模型
    model.learn(total_timesteps=10000, callback=checkpoint_callback)
    print(f"model learn finish: {model} -> {name_prefix}")

    # 评估模型
    model.save(f'./logs/{name_prefix}')  # 保存模型
    del model  # 删除模型以便重新加载

    # 关闭环境
    env.close()
