import os  # 用于处理文件和目录的工具
import sys  # 提供对Python解释器使用或维护的变量和函数的访问
import random  # 提供生成随机数的功能

import torch  # PyTorch库，用于深度学习
from stable_baselines3.common.monitor import Monitor  # 监控环境的工具
from stable_baselines3.common.vec_env import SubprocVecEnv  # 用于处理多个环境的并行化
from stable_baselines3.common.callbacks import CheckpointCallback  # 用于保存模型检查点的回调函数

from sb3_contrib import MaskablePPO  # 引入MaskablePPO算法（带有动作掩蔽的PPO）
from sb3_contrib.common.wrappers import ActionMasker  # 用于在环境中应用动作掩蔽的工具

from snake_game_custom_wrapper_cnn import SnakeEnv  # 自定义的贪吃蛇游戏环境

# 根据是否支持MPS（Metal Performance Shaders）来设置环境数量
# MPS（Metal Performance Shaders）是苹果公司提供的一组高性能计算框架，专门用于在其设备（如Mac和iOS设备）上加速机器学习和图像处理任务。
# 如果你在使用PyTorch框架进行深度学习或其他计算密集型任务时，可以利用这些功能来加速计算。
if torch.backends.mps.is_available():
    NUM_ENV = 32 * 2  # 如果支持MPS，则使用64个环境
else:
    NUM_ENV = 32  # 否则使用32个环境

LOG_DIR = "logs"  # 定义日志目录

os.makedirs(LOG_DIR, exist_ok=True)  # 创建日志目录，如果已经存在则不报错


# 线性调度器函数，用于调整学习率等参数
def linear_schedule(initial_value, final_value=0.0):
    if isinstance(initial_value, str):  # 如果初始值是字符串，则转换为浮点数
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert (initial_value > 0.0)  # 确保初始值大于0

    # 调度器函数，根据进度返回当前值
    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler  # 返回调度器函数


# 创建环境的工厂函数
def make_env(seed=0):
    def _init():  # 初始化环境
        env = SnakeEnv(seed=seed)  # 创建贪吃蛇环境
        env = ActionMasker(env, SnakeEnv.get_action_mask)  # 应用动作掩蔽
        env = Monitor(env)  # 监控环境
        env.seed(seed)  # 设置环境随机种子
        return env  # 返回环境
    return _init  # 返回初始化函数


def main():
    # 生成一组随机种子
    seed_set = set()
    while len(seed_set) < NUM_ENV:  # 确保种子数量等于环境数量
        seed_set.add(random.randint(0, 1e9))  # 生成随机种子并存储在集合中

    # 创建并行的贪吃蛇环境
    env = SubprocVecEnv([make_env(seed=s) for s in seed_set])

    # 根据是否支持MPS选择学习率和创建模型
    if torch.backends.mps.is_available():
        lr_schedule = linear_schedule(5e-4, 2.5e-6)  # 设置学习率调度
        clip_range_schedule = linear_schedule(0.150, 0.025)  # 设置剪切范围调度
        # 实例化一个PPO代理，使用MPS
        model = MaskablePPO(
            "CnnPolicy",  # 使用卷积神经网络策略
            env,  # 设置环境
            device="mps",  # 使用MPS设备
            verbose=1,  # 输出详细信息
            n_steps=2048,  # 每次更新的步数
            batch_size=512 * 8,  # 批处理大小
            n_epochs=4,  # 训练的轮数
            gamma=0.94,  # 折扣因子
            learning_rate=lr_schedule,  # 学习率调度
            clip_range=clip_range_schedule,  # 剪切范围调度
            tensorboard_log=LOG_DIR  # 日志目录
        )
    else:
        lr_schedule = linear_schedule(2.5e-4, 2.5e-6)  # 设置学习率调度
        clip_range_schedule = linear_schedule(0.150, 0.025)  # 设置剪切范围调度
        # 实例化一个PPO代理，使用CUDA
        model = MaskablePPO(
            "CnnPolicy",
            env,
            device="cuda",  # 使用CUDA设备
            verbose=1,
            n_steps=2048,
            batch_size=512,  # 批处理大小
            n_epochs=4,
            gamma=0.94,
            learning_rate=lr_schedule,
            clip_range=clip_range_schedule,
            tensorboard_log=LOG_DIR
        )

    # 设置模型保存目录
    if torch.backends.mps.is_available():
        save_dir = "trained_models_cnn_mps"  # MPS设备的模型保存目录
    else:
        save_dir = "trained_models_cnn"  # CUDA设备的模型保存目录
    os.makedirs(save_dir, exist_ok=True)  # 创建保存目录

    checkpoint_interval = 15625  # 每15625步保存一次模型
    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_interval, save_path=save_dir, name_prefix="ppo_snake")  # 创建检查点回调

    # 将训练日志从stdout重定向到文件
    original_stdout = sys.stdout  # 保存原始stdout
    log_file_path = os.path.join(save_dir, "training_log.txt")  # 定义日志文件路径
    with open(log_file_path, 'w') as log_file:  # 打开日志文件
        sys.stdout = log_file  # 将stdout重定向到日志文件

        # 开始训练
        model.learn(
            total_timesteps=int(100000000),  # 训练总步数
            callback=[checkpoint_callback]  # 添加检查点回调
        )
        env.close()  # 关闭环境

    # 恢复stdout
    sys.stdout = original_stdout

    # 保存最终模型
    model.save(os.path.join(save_dir, "ppo_snake_final.zip"))  # 将模型保存到指定路径


# 程序入口
if __name__ == "__main__":
    main()  # 调用主函数
