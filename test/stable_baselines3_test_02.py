import gym
import os
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker  # 用于在环境中应用动作掩蔽的工具
import common

# game_name = 'CartPole-v1'
game_name = 'MountainCar-v0'
# game_name = 'game_gridworld'
model_file = f"{game_name.replace('-','_').lower()}_test02"

if __name__ == '__main__':
    print("start pid: %d" % (os.getpid()))
    env, _ = common.gym_make(game_name)

    # 创建 PPO 模型
    # env = Monitor(env)
    print(f"动作空间的形状: {env.action_space.shape}, 可能的动作数量: {env.action_space.n}")
    print(f"观察数据的形状: {env.observation_space.shape}")

    # CnnPolicy
    # model = PPO('MlpPolicy', env, verbose=1)
    model = MaskablePPO('MlpPolicy', env=env,
                        device="cuda",
                        verbose=1,
                        n_steps=2048,
                        batch_size=512,
                        n_epochs=4,
                        gamma=0.94)
    # model = PPO('MultiInputPolicy', env, verbose=1)
    # os._exit(1)

    # 定义检查点回调
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/',
                                             name_prefix=model_file)

    # 训练模型
    model.learn(total_timesteps=10000, callback=checkpoint_callback, use_masking=True)
    print(f"model learn finish: {model} -> {model_file}")

    # 评估模型
    model.save(f'./logs/{model_file}')  # 保存模型
    del model  # 删除模型以便重新加载

    # 关闭环境
    env.close()
