from game_gridworld import *
import math
import os
import logging
from sb3_contrib.common.wrappers import ActionMasker  # 用于在环境中应用动作掩蔽的工具


def mountaincar_reward(env, obs, rewards, done, info):
    # 胜利判断
    if done and rewards >= 1:
        return obs, 1000, done, info

    rewards = 0
    values = obs.reshape(-1)
    distance = values[0]  # -1.1 ~ 0.1, -0.5为起点
    speed = values[1]  # 0~1
    # speed = math.sqrt(env.observation_space.low[0]**2 + env.observation_space.low[1] ** 2)
    # print(f"x: {env.observation_space.high} speed: {env.observation_space.low}")
    # print(f"pos: {distance} speed: {speed}")

    # 策略1, 速度越大越好(绝对值)
    speed_reward = abs(speed)  # 0 ~ 0.05

    # 策略2, 越高越好
    distance_reward = abs(distance - -0.5)  # 0 ~ 1

    # 策略3, 时间惩罚
    time_punish = (env._step_times / 200)  # 0 ~ 1

    # 策略3: 最大奖励刷新, 在最大奖励的基础上增加数值
    env._max_rewards = max(speed_reward + distance_reward, float(env._max_rewards))

    # 计算
    # rewards = env._max_rewards * 3 + speed_reward + distance_reward - time_punish
    rewards = ((speed_reward * 20 + distance_reward) * 0.5) - time_punish

    if env._step_times % 1000 == 0:
        print(f"[{env._step_times}] obs: {obs.reshape(-1)} rewards: {rewards}, done: {done} info: {info}")

    return obs, rewards, done, info


def mountaincar_check_action_validity(env, action: int) -> bool:
    # print(f"mountaincar_check_action_validity: {action} -> { action != 0}")
    return action != 0


class GameEnv(gym.Wrapper):
    def __init__(self, env, reward_func=None, _check_action_validity=None):
        super().__init__(env=env)
        self._reward_func = reward_func
        self._check_action_validity = _check_action_validity
        self._step_times = 0
        self._max_rewards = 0

    def reset(self, **kwargs):
        self._step_times = 0
        self._max_rewards = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        self._step_times += 1
        if self._reward_func is None:
            return self.env.step(action)
        # 使用_reward_func
        obs, rewards, done, info = self.env.step(action)
        return self._reward_func(self, obs, rewards, done, info)

    def get_action_mask(self):
        if self._check_action_validity is None:
            return np.array([[True for _ in range(self.action_space.n)]])

        # 获取有效动作掩码
        return np.array([[self._check_action_validity(self, a) for a in range(self.action_space.n)]])


def gym_make(game_name):
    if game_name == "game_gridworld":
        return ActionMasker(env=GridWorldEnv(), action_mask_fn=GridWorldEnv.get_action_mask), 1
    elif game_name == 'MountainCar-v0':
        # https://zhuanlan.zhihu.com/p/599570548
        # https://blog.csdn.net/qq_43674552/article/details/130616241
        e = GameEnv(gym.make(game_name), mountaincar_reward, mountaincar_check_action_validity)
        return ActionMasker(env=e, action_mask_fn=GameEnv.get_action_mask), 30
    return gym.make(game_name), 30




# 在文件开头添加logger配置
def setup_logger(name, log_file, level=logging.INFO):
    """设置logger"""
    # 创建logs目录（如果不存在）
    os.makedirs('logs', exist_ok=True)
    
    # 创建格式化器
    formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 创建处理器 - 文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # 创建处理器 - 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # 获取logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 清除已存在的处理器
    logger.handlers.clear()
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger