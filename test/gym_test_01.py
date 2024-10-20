
import gym

env = gym.make('CartPole-v1')  # 创建一个 CartPole 环境(游戏环境)

obs = env.reset()  # 重置环境并获得初始观察
for _ in range(1000):
    action = env.action_space.sample()  # 随机选择一个动作
    obs, reward, done, info = env.step(action)  # 执行动作
    env.render()  # 可视化环境

    if done:
        obs = env.reset()  # 如果结束，重置环境


env.close()  # 关闭环境，释放资源
