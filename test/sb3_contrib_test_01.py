import gym
from gym import spaces
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker


# 自定义环境类,继承自gym.Env
class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # 定义动作空间为离散空间,有5个可能的动作
        self.action_space = spaces.Discrete(5)
        # 定义观察空间为一维连续空间,取值范围为[0, 1]
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.state = 0  # 初始化状态
        self.steps = 0  # 初始化步数计数器

    def reset(self):
        # 重置环境:随机生成新的状态,重置步数,并返回初始观察
        self.state = np.random.rand()
        self.steps = 0
        return np.array([self.state])

    def step(self, action):
        self.steps += 1  # 增加步数
        # 如果动作等于状态值乘5的整数部分,奖励为1,否则为-1
        reward = 1 if action == int(self.state * 5) else -1
        done = self.steps >= 10  # 如果步数达到10,则结束回合
        self.state = np.random.rand()  # 生成新的随机状态
        return np.array([self.state]), reward, done, {}

    def get_action_mask(self):
        # 生成动作掩码:这里假设只有前两个动作是合法的
        return [True, True, False, False, False]


# 定义动作掩蔽函数
def mask_fn(env):
    return env.get_action_mask()


# 创建自定义环境实例
env = CustomEnv()

# 使用ActionMasker包装环境,应用动作掩蔽
env = ActionMasker(env, mask_fn)

# 创建MaskablePPO模型,使用默认的MLP策略网络
model = MaskablePPO("MlpPolicy", env, verbose=1)

# 训练模型,总时间步数为10000
model.learn(total_timesteps=10000)

# 测试训练好的模型
obs = env.reset()  # 重置环境,获取初始观察
for i in range(10):  # 进行10个步骤的测试
    # 使用模型预测下一个动作
    action, _states = model.predict(obs, deterministic=True)
    # 执行动作,获取新的观察、奖励、是否结束等信息
    obs, rewards, done, info = env.step(action)
    # 打印每一步的信息
    print(f"Step {i+1}, Action: {action}, Reward: {rewards}, Done: {done}")

    if done:  # 如果回合结束,重置环境
        obs = env.reset()
