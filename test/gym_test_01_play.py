import gym
import pygame
import numpy as np
import time
print(gym.envs.registry.all())
# CartPole-v1 部署成磁盘输入

# 初始化 pygame
pygame.init()

# 创建 Gym 环境
# env = gym.make('CartPole-v1')  # CartPole-v1（平衡车）
# action_map = {
#     pygame.K_LEFT: 0,
#     pygame.K_RIGHT: 1,
#     0: None,  # 默认
# }

env = gym.make('MountainCar-v0')  # MountainCar-v0（山地车）
action_map = {
    pygame.K_LEFT: 0,  # 向左推
    pygame.K_RIGHT: 2,  # 向右推
    0: 1,  # 默认
} 

# 设置窗口
screen = pygame.display.set_mode((600, 400))
pygame.display.set_caption("CartPole Control")

# 重置环境
observation = env.reset()

print(f"动作空间的形状: {env.action_space.shape}")
print(f"可能的动作数量: {env.action_space.n}")
# 打印每个可能的动作
print("可能的动作:")
for action in range(env.action_space.n):
    print(f"  动作 {action}: {env.action_space.contains(action)}")

# 设置时钟
clock = pygame.time.Clock()

# 游戏循环
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 获取键盘输入
    keys = pygame.key.get_pressed()
    action = next((action for key, action in action_map.items() if keys[key]), action_map[0])  # 默认动作为 1

    done = None
    if action != None:
        print("action: %d" % (action))
        # 采取动作
        observation, reward, done, info = env.step(action)

        # # 渲染环境
        # frame = env.render()
        # frame = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        # screen.blit(pygame.transform.scale(frame, (600, 400)), (0, 0))
        # pygame.display.flip()
    else:
        continue

    if done:
        print("game is done to reset")
        observation = env.reset()
        time.sleep(1)

    # 渲染环境
    env.render()

    # 控制帧率
    clock.tick(60)
    # time.sleep(0.1)

# 结束
env.close()
pygame.quit()
