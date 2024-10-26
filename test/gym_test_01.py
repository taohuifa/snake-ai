import gym

env = gym.make('CartPole-v1')  # 创建一个 CartPole 环境(游戏环境)

obs = env.reset()  # 重置环境并获得初始观察

# 打印动作空间的详细信息
print(f"动作空间类型: {type(env.action_space)}")
print(f"动作空间: {env.action_space}")
print(f"动作空间的形状: {env.action_space.shape}")
print(f"可能的动作数量: {env.action_space.n}")

# 打印每个可能的动作
print("可能的动作:")
for action in range(env.action_space.n):
    print(f"  动作 {action}: {env.action_space.contains(action)}")

for _ in range(1000):
    action = env.action_space.sample()  # 随机选择一个动作
    # print(f"选择的动作: {action}")
    obs, reward, done, info = env.step(action)  # 执行动作
    env.render()  # 可视化环境

    if done:
        obs = env.reset()  # 如果结束，重置环境
        break

env.close()  # 关闭环境，释放资源
