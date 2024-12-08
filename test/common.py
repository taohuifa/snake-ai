from game_gridworld import *
import math
import os
import logging
from sb3_contrib.common.wrappers import ActionMasker  # 用于在环境中应用动作掩蔽的工具


def adjustme(next_obs):
    x = next_obs[0]     # 小车位置，范围在 -1.2 到 0.6 之间
    speed = next_obs[1]  # 小车速度，范围在 -0.07 到 0.07 之间
    reward = 0

    # 加上速度
    reward += abs(speed) * 10
    # 目标区域奖励：当小车位置 > 0.1 时
    if x > -0.5:
        # 基础奖励10分，距离目标越近额外奖励越高
        # 比如：位置在0.2时，奖励为 10 + (0.2-0.1)*100 = 20分
        reward += (10.0 + (x - -0.5) * 100) / 50

    # 蓄力区域奖励：当小车位置在 -0.9 到 -0.7 之间
    elif x <= -0.5:
        # 基础奖励6分，越靠近-0.9位置额外奖励越高
        # 比如：位置在-0.8时，奖励为 6 + (-0.7-(-0.8))*100 = 16分
        reward += (6.0 + (-0.5 - x) * 100) / 50

    # 其他区域都给予-1的负奖励
    else:
        reward = -1

    return reward


def mountaincar_reward(env, obs, rewards, done, info):
    values = obs.reshape(-1)
    distance = values[0]  # -1.1 ~ 0.1, -0.5为起点
    speed = values[1]  # 0~1

    # 胜利判断
    if done:
        rewards = 1 if distance > 0.5 else -1
        print(f"[{env._step_times}] done obs: {obs.reshape(-1)} rewards: {rewards}, done: {done} info: {info}")
        return obs, rewards, done, info

    if env._step_times >= 150:
        done = True  # 强制结束

    # 这是一个连续的空间，包括小车的位置和速度。状态由两个数值表示：
    # 位置 xxx：范围在 -1.2 到 0.6 之间。
    # 速度 vvv：范围在 -0.07 到 0.07 之间。

    rewards = 0
    # # speed = math.sqrt(env.observation_space.low[0]**2 + env.observation_space.low[1] ** 2)
    # # print(f"x: {env.observation_space.high} speed: {env.observation_space.low}")
    # # print(f"pos: {distance} speed: {speed}")

    # 策略1, 速度越大越好(绝对值)
    speed_reward = abs(speed) * 10  # 0 ~ 0.05
    # speed_reward = speed**2 * 0.5

    # 策略2, 越高越好
    distance_reward = abs(distance - -0.5)  # 0 ~ 1
    # distance_reward = (distance - -0.5) ** 2 * 0.5
    # distance_reward = 0

    # 策略3, 时间惩罚
    # time_punish = (env._step_times / 200) * 0.1  # 0 ~ 1
    time_punish = 0

    # # 策略3: 最大奖励刷新, 在最大奖励的基础上增加数值
    # env._max_rewards = max(speed_reward + distance_reward, float(env._max_rewards))

    # # 计算
    # # rewards = env._max_rewards * 3 + speed_reward + distance_reward - time_punish
    # rewards = ((speed_reward * 20 + distance_reward) * 0.5) - time_punish - 0.5
    cur_rewards = distance_reward + speed_reward - time_punish

    # 根据增长, 计算奖励
    # rewards = (cur_rewards - float(env._pre_cur_rewards)) /float(env._pre_cur_rewards)
    # rewards = np.clip(rewards, -1, 1)
    rewards = cur_rewards - 1

    # rewards = adjustme(obs)
    # print(f"{env._pre_cur_rewards} -> {cur_rewards} = {rewards}")
    env._pre_cur_rewards = cur_rewards
    # rewards = cur_rewards

    if env._step_times % 10000 == 0:
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
        self.reset()

    def reset(self, **kwargs):
        self._step_times = 0
        self._max_rewards = 0
        self._pre_obs = None
        self._obs = None
        self._rewards = 0
        self._pre_reward = 0
        self._pre_cur_rewards = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        self._step_times += 1
        self._pre_obs = self._obs
        self._pre_reward = self._rewards
        if self._reward_func is None:
            return self.env.step(action)
        # 使用_reward_func
        obs, rewards, done, info = self.env.step(action)
        self._obs = obs
        self._rewards = rewards
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


from pynput import keyboard

current_key = None
current_action = None
_actions = {}


def on_press(key):
    global current_key, current_action
    current_key = key
    try:
        current_action = _actions[key]
        print(f"press: {key} -> {current_action}")
        # 0: 上, 1: 左, 2: 右, 3: 下 : keyboard.Key.up
    except Exception:
        # pass
        print(f"no found press key: {key}")


def on_release(key):
    global current_key, current_action
    current_key = None
    current_action = _actions[0]
    print(f"release key: {key} -> {current_action}")


def init_keyboard_listener(actions):
    global _actions, current_action
    _actions = actions
    current_action = _actions[0]
    print(f"keyboard: {_actions}")

    # 在主函数开始处添加监听器
    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release)
    listener.start()
