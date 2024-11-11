import gym  # 导入OpenAI Gym库，用于创建和管理环境
import numpy as np  # 导入NumPy库，用于数值计算
import pygame
from gym import spaces


class GridWorldEnv(gym.Env):
    metadata = {"render_fps": 4}  # 这个metadata用于保存render（启动画面，下文会提到）的参数

    def __init__(self, size=5, render_mode="none"):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.np_random = np.random.default_rng()  # 添加这一行以初始化 np_random

        # Obervations are dictionaries with the agent's and the target's loaction
        # Each location is encoded as an element of {0,...,'size'}^2,i.e. MultiDiscrete([size,size])
        self.observation_space = spaces.Box(low=0, high=1, shape=(2, 2,), dtype=np.float32)  # 示例
        # We have 4 actions,corresponding to "right,left,up,down"
        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1])
        }

        self.step_idx = 0
        self.max_step = size * size
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.reset()

    def _get_obs(self):
        # 返回一个二维数组，包含代理和目标的位置
        return np.array([[self._agent_location[0], self._agent_location[1]],
                         [self._target_location[0], self._target_location[1]]],
                        dtype=np.float32)

    def _get_info(self):
        return {}  # 返回一个空字典，您可以根据需要添加更多信息

    def reset(self, options=None):
        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agents's location
        self._target_location = self._agent_location
        while np.array_equal(self._agent_location, self._target_location):
            self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        self.step_idx = 0
        objs = self._get_obs()
        # info = self._get_info()

        self._render_frame(self.render_mode)
        return objs

    def step(self, action, extra_info=None):
        # 确保 action 是整数
        action = int(action)  # 添加这一行以确保 action 是整数
        direction = self._action_to_direction[action]
        # We use np.clip to make sure we don't leave the grid
        self._agent_location = np.clip(self._agent_location + direction, 0, self.size - 1)
        self.step_idx += 1

        # 使用 np.where 简化 terminated 的计算
        terminated = np.array_equal(self._agent_location, self._target_location)

        # 合并 reward 的计算逻辑
        reward = 1 if terminated else -1 if self.step_idx >= self.max_step else 0
        terminated = terminated or (self.step_idx >= self.max_step)

        observation = self._get_obs()
        info = self._get_info()

        self._render_frame(self.render_mode)
        return observation, reward, terminated, info

    def close():
        return

    def seed():
        return

    def render(self, mode: str = "human"):
        return self._render_frame(mode)

    def _render_frame(self, mode: str = "human"):
        if mode is None or mode == "none":
            return
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        pix_square_size = (self.window_size / self.size)  # Size of single grid square in pixels

        # First we draw the target
        pygame.draw.rect(canvas, (255, 0, 0), pygame.Rect(pix_square_size * self._target_location, (pix_square_size, pix_square_size)))
        # Then draw the agent
        pygame.draw.circle(canvas, (0, 0, 255), (self._agent_location + 0.5) * pix_square_size, pix_square_size / 3)

        # Finally,add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas, 0, (0, pix_square_size * x), (self.window_size, pix_square_size * x), width=3
            )
            pygame.draw.line(
                canvas,
                0, (pix_square_size * x, 0), (pix_square_size * x, self.window_size), width=3
            )

        if mode == "human":
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((self.window_size, self.window_size))
            if self.clock is None:
                self.clock = pygame.time.Clock()
                # The following line copies our drawing from 'canvas' to the visible window

            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable
            # self.clock.tick(self.metadata["render_fps"]) # 不进行频率等待
            return

        # rgb_array
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )
