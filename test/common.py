from game_gridworld import *
import math
from sb3_contrib.common.wrappers import ActionMasker  # 用于在环境中应用动作掩蔽的工具


def mountaincar_reward(env, obs, rewards, done, info):
    # 胜利判断
    if done and rewards >= 1:
        return obs, 1000, done, info

    values = obs.reshape(-1)
    # rewards = 1 - (0.52089536 - values[0])
    # rewards = 1 - math.sqrt((0.50974405 - values[0]) ** 2 + (0.03367784 - values[1]) ** 2)
    rewards = math.sqrt((values[0] - -0.56914455)**2 + (values[1] - 0.00069396)**2) - 0.1
    # rewards = math.sqrt(values[0] ** 2 + values[1] ** 2)
    rewards = rewards - (env.step_times / 200)

    if env.step_times % 1000 == 0:
        print(f"[{env.step_times}] obs: {obs.reshape(-1)} rewards: {rewards}, done: {done} info: {info}")

    return obs, rewards, done, info


def mountaincar_check_action_validity(env, action: int) -> bool:
    # print(f"mountaincar_check_action_validity: {action} -> { action != 0}")
    return action != 0


class GameEnv(gym.Wrapper):
    def __init__(self, env, reward_func=None, _check_action_validity=None):
        super().__init__(env=env)
        self._reward_func = reward_func
        self._check_action_validity = _check_action_validity
        self.step_times = 0

    def reset(self, **kwargs):
        self.step_times = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        self.step_times += 1
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
        return GridWorldEnv(), 1
    elif game_name == 'MountainCar-v0':
        e = GameEnv(gym.make(game_name), mountaincar_reward, mountaincar_check_action_validity)
        return ActionMasker(env=e, action_mask_fn=GameEnv.get_action_mask), 30
    return gym.make(game_name), 30
