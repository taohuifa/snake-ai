import gym
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback


# 定义环境创建函数
def make_env(idx):
    def _init():
        print("init_monitor: %d, pid: %d from: %d" % (idx, os.getpid(), os.getppid()))
        return Monitor(gym.make('CartPole-v1'))
    return _init


if __name__ == '__main__':
    print("start pid: %d" % (os.getpid()))
    # 创建多个并行环境
    num_envs = 2  # 并行环境数量
    envs = SubprocVecEnv([make_env(i) for i in range(num_envs)])

    # 创建 PPO 模型
    model = PPO('MlpPolicy', envs, verbose=1)

    # 定义检查点回调
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/',
                                             name_prefix='ppo_cartpole')

    # 训练模型
    model.learn(total_timesteps=10000, callback=checkpoint_callback)

    # 关闭环境
    envs.close()
