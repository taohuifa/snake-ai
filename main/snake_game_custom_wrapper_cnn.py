import math

import gym
import numpy as np

from snake_game import SnakeGame

class SnakeEnv(gym.Env):
    def __init__(self, seed=0, board_size=12, silent_mode=True, limit_step=True):
        super().__init__()
        # 初始化贪吃蛇游戏
        self.game = SnakeGame(seed=seed, board_size=board_size, silent_mode=silent_mode)
        self.game.reset()

        self.silent_mode = silent_mode

        # 定义动作空间: 0: 上, 1: 左, 2: 右, 3: 下
        self.action_space = gym.spaces.Discrete(4)
        
        # 定义观察空间: 84x84x3 的RGB图像
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(84, 84, 3),
            dtype=np.uint8
        )

        self.board_size = board_size
        self.grid_size = board_size ** 2  # 蛇的最大长度为棋盘大小的平方
        self.init_snake_size = len(self.game.snake)
        self.max_growth = self.grid_size - self.init_snake_size

        self.done = False

        # 设置步数限制
        if limit_step:
            self.step_limit = self.grid_size * 4  # 足够获取食物的步数
        else:
            self.step_limit = 1e9  # 基本上没有限制

        self.reward_step_counter = 0

    def reset(self):
        # 重置游戏状态
        self.game.reset()

        self.done = False
        self.reward_step_counter = 0

        obs = self._generate_observation()
        return obs
    
    def step(self, action):
        # 执行动作并获取游戏状态
        self.done, info = self.game.step(action)
        obs = self._generate_observation()

        reward = 0.0
        self.reward_step_counter += 1

        # 蛇填满整个棋盘，游戏胜利
        if info["snake_size"] == self.grid_size:
            reward = self.max_growth * 0.1  # 胜利奖励
            self.done = True
            if not self.silent_mode:
                self.game.sound_victory.play()
            return obs, reward, self.done, info
        
        # 达到步数限制，游戏结束
        if self.reward_step_counter > self.step_limit:
            self.reward_step_counter = 0
            self.done = True
        
        # 蛇撞墙或撞到自己，游戏结束
        if self.done:
            # 游戏结束惩罚基于蛇的大小
            reward = - math.pow(self.max_growth, (self.grid_size - info["snake_size"]) / self.max_growth) * 0.1
            return obs, reward, self.done, info
          
        # 蛇吃到食物
        elif info["food_obtained"]:
            reward = info["snake_size"] / self.grid_size
            self.reward_step_counter = 0  # 重置奖励步数计数器
        
        else:
            # 根据蛇是否朝食物方向移动给予小额奖励/惩罚
            if np.linalg.norm(info["snake_head_pos"] - info["food_pos"]) < np.linalg.norm(info["prev_snake_head_pos"] - info["food_pos"]):
                reward = 1 / info["snake_size"]
            else:
                reward = - 1 / info["snake_size"]
            reward = reward * 0.1

        # max_score: 72 + 14.1 = 86.1
        # min_score: -14.1
        return obs, reward, self.done, info
    
    def render(self):
        self.game.render()

    def get_action_mask(self):
        # 获取有效动作掩码
        return np.array([[self._check_action_validity(a) for a in range(self.action_space.n)]])
    
    def _check_action_validity(self, action):
        # 检查动作是否有效（不会导致游戏立即结束）
        current_direction = self.game.direction
        snake_list = self.game.snake
        row, col = snake_list[0]

        # 根据动作更新蛇头位置
        if action == 0:  # 上
            if current_direction == "DOWN":
                return False
            else:
                row -= 1
        elif action == 1:  # 左
            if current_direction == "RIGHT":
                return False
            else:
                col -= 1
        elif action == 2:  # 右
            if current_direction == "LEFT":
                return False
            else:
                col += 1     
        elif action == 3:  # 下
            if current_direction == "UP":
                return False
            else:
                row += 1

        # 检查蛇是否撞到自己或墙壁
        if (row, col) == self.game.food:
            game_over = (
                (row, col) in snake_list
                or row < 0
                or row >= self.board_size
                or col < 0
                or col >= self.board_size
            )
        else:
            game_over = (
                (row, col) in snake_list[:-1]
                or row < 0
                or row >= self.board_size
                or col < 0
                or col >= self.board_size
            )

        if game_over:
            return False
        else:
            return True

    def _generate_observation(self):
        # 生成观察状态
        # 空白: 黑色; 蛇身: 灰色; 蛇头: 绿色; 食物: 红色
        obs = np.zeros((self.game.board_size, self.game.board_size), dtype=np.uint8)

        # 设置蛇身为灰色，从头到尾线性递减强度
        obs[tuple(np.transpose(self.game.snake))] = np.linspace(200, 50, len(self.game.snake), dtype=np.uint8)
        
        # 将单层堆叠成3通道图像
        obs = np.stack((obs, obs, obs), axis=-1)
        
        # 设置蛇头为绿色，蛇尾为蓝色
        obs[tuple(self.game.snake[0])] = [0, 255, 0]
        obs[tuple(self.game.snake[-1])] = [255, 0, 0]

        # 设置食物为红色
        obs[self.game.food] = [0, 0, 255]

        # 将观察状态放大到84x84
        obs = np.repeat(np.repeat(obs, 7, axis=0), 7, axis=1)

        return obs

# 使用随机动作测试环境的代码（已注释掉）
# NUM_EPISODES = 100
# RENDER_DELAY = 0.001
# from matplotlib import pyplot as plt

# if __name__ == "__main__":
#     env = SnakeEnv(silent_mode=False)
    
    # # 测试初始化效率
    # print(MODEL_PATH_S)
    # print(MODEL_PATH_L)
    # num_success = 0
    # for i in range(NUM_EPISODES):
    #     num_success += env.reset()
    # print(f"Success rate: {num_success/NUM_EPISODES}")

    # sum_reward = 0

    # # 0: 上, 1: 左, 2: 右, 3: 下
    # action_list = [1, 1, 1, 0, 0, 0, 2, 2, 2, 3, 3, 3]
    
    # for _ in range(NUM_EPISODES):
    #     obs = env.reset()
    #     done = False
    #     i = 0
    #     while not done:
    #         plt.imshow(obs, interpolation='nearest')
    #         plt.show()
    #         action = env.action_space.sample()
    #         # action = action_list[i]
    #         i = (i + 1) % len(action_list)
    #         obs, reward, done, info = env.step(action)
    #         sum_reward += reward
    #         if np.absolute(reward) > 0.001:
    #             print(reward)
    #         env.render()
            
    #         time.sleep(RENDER_DELAY)
    #     # print(info["snake_length"])
    #     # print(info["food_pos"])
    #     # print(obs)
    #     print("sum_reward: %f" % sum_reward)
    #     print("episode done")
    #     # time.sleep(100)
    
    # env.close()
    # print("Average episode reward for random strategy: {}".format(sum_reward/NUM_EPISODES))
