"""
MountainCar-v0 实验编号 01: MaskablePPO + reward v01

===========================================================================
文件结构（方便复制改造，展开多种模型对比）:
  1. 配置区        — 超参数集中管理
  2. Reward 函数   — 可自定义，每个文件独立改造
  3. 环境构建      — 复用 common.GameEnv，传入自定义 reward
  4. 评估回调      — 训练中自动评估 + 保存最佳模型
  5. 训练 / 测试   — --mode train|play 切换

复制本文件为 maskable_ppo_02.py 即可开始新实验，MODEL_FILE 自动跟随文件名。
===========================================================================

Reward v01 设计（修复原 common.py 的核心问题）:
  原代码计算了 speed_reward 和 distance_reward，但最终只用 height-0.5，
  速度信号完全丢失，导致 agent 无法学到"蓄力摆动"策略。
  v01 三合一: 高度 + 速度 + 位置进展。

用法:
  训练:  python test/mountainCar_v0/maskable_ppo_01.py --mode train
  测试:  python test/mountainCar_v0/maskable_ppo_01.py --mode play
"""
import sys
import os
import time
import datetime
import argparse
import numpy as np
import gym
import pygame

# 复用 test/common.py 的 GameEnv / setup_logger（不改动 common.py）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import common
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# ============================ 配置区 ============================
GAME = 'MountainCar-v0'
# 用脚本名作为模型标识，复制文件即自动区分
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
MODEL_FILE = f"mountaincar_v0_{SCRIPT_NAME}"
RENDER_TICK = 30

# ---- 训练超参数 ----
PARAMS = {
    'total_timesteps': 200000,
    'gamma':           0.99,    # 折扣因子，重视长期回报
    'ent_coef':        0.01,    # 熵正则，鼓励探索
    'n_epochs':        10,
    'batch_size':      256,
    'n_steps':         2048,
    'learning_rate':   3e-4,
    'clip_range':      0.2,
    'gae_lambda':      0.95,
    'device':          'cuda',
    'verbose':         1,
    'eval_freq':       20000,
    'n_eval_episodes': 10,
}

# ---- 观测归一化 (VecNormalize) ----
# MountainCar 两个观测值量级差异极大:
#   position: [-1.2, 0.6]    量级 ~1
#   velocity: [-0.07, 0.07]  量级 ~0.01
# 不归一化时神经网络严重偏向 position，velocity 信号几乎被忽略。
# VecNormalize 在训练中动态统计均值/方差，将观测归一化到 ~N(0,1)。
# 这是 SB3 官方 RL Zoo 通关 MountainCar-v0 的关键步骤。
# 参考: https://huggingface.co/sb3/ppo-MountainCar-v0 (normalize: true)
NORM_OBS = True


# ============================ Reward 函数（可自定义）============================
def _height(xs):
    """山地高度函数，与 MountainCar 内部物理一致"""
    return np.sin(3 * xs) * .45 + .55


def reward_v01(env, obs, rewards, done, info):
    """
    v01 Reward: 高度 + 速度 + 位置进展

    修复原 common.py 的核心问题：
      原代码计算了 speed_reward / distance_reward，但最终 rewards = height - 0.5，
      速度信号完全丢失，agent 无法学到"蓄力摆动"策略。

    三合一设计:
      height_r  — 高度信号，越高越好 [-0.4, 0.5]
      speed_r   — 速度绝对值，鼓励蓄力 [0, 0.7]
      progress  — 相对起点(-0.5)的位置进展，引导向右 [-0.35, 0.55]
    """
    pos = obs.reshape(-1)[0]   # [-1.2, 0.6]
    vel = obs.reshape(-1)[1]  # [-0.07, 0.07]

    # 终局奖励
    if done:
        r = 1.0 if pos > 0.5 else -1.0
        return obs, r, done, info

    # 200 步截断（原 common.py 150 太短）
    if env._step_times >= 500:
        done = True

    # --- 三合一 reward shaping ---
    height_r = _height(pos) - 0.5       # 高度信号
    speed_r = abs(vel) * 10             # 速度信号（修复：原代码计算了但没用）
    progress_r = (pos + 0.5) * 0.5      # 位置进展，引导向右

    rewards = height_r + speed_r + progress_r
    return obs, rewards, done, info


def check_action_validity(env, action: int) -> bool:
    """
    动作有效性检查

    v01: 全放行（不限制 action 1）
    MountainCar 最优策略有时需要"不推"利用惯性，禁止会降低探索效率。
    如需实验 mask 效果，改为 `return action != 1`。
    """
    return True


# ============================ 环境构建 ============================
def make_env(reward_func=None, check_func=None):
    """
    构建 MountainCar 环境，复用 common.GameEnv

    参数:
        reward_func: 自定义 reward，None 用 reward_v01
        check_func:  自定义 action mask，None 用 check_action_validity
    返回:
        (env, tick): 封装后的环境 + 渲染帧率
    """
    rf = reward_func if reward_func is not None else reward_v01
    cf = check_func if check_func is not None else check_action_validity
    base = gym.make(GAME)
    e = common.GameEnv(base, rf, cf)
    env = ActionMasker(env=e, action_mask_fn=common.GameEnv.get_action_mask)
    return env, RENDER_TICK


# ============================ 评估回调 ============================
class EvalCallback(BaseCallback):
    """每 N 步评估，保存最佳模型"""

    def __init__(self, eval_env, eval_freq, n_eval_episodes, save_prefix,
                 train_vec_normalize=None, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_prefix = save_prefix
        self.best_mean_reward = -np.inf
        # 训练环境的 VecNormalize，用于同步归一化统计量到评估环境
        self.train_vec_normalize = train_vec_normalize

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            # 评估前同步归一化统计量（评估环境用训练环境的统计来归一化观测）
            if self.train_vec_normalize is not None:
                self.eval_env.obs_rms = self.train_vec_normalize.obs_rms
            mean_r, std_r = evaluate_policy(
                self.model, self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True,
                use_masking=True
            )
            print(f"\n[Eval @ {self.n_calls}] mean_reward: {mean_r:.2f} +/- {std_r:.2f}")
            if mean_r > self.best_mean_reward:
                self.best_mean_reward = mean_r
                self.model.save(f"logs/{self.save_prefix}_best")
                print(f"  >> 新最佳模型: logs/{self.save_prefix}_best (reward={mean_r:.2f})")
        return True


# ============================ 训练 ============================
def train():
    logger = common.setup_logger(
        SCRIPT_NAME,
        f'logs/{SCRIPT_NAME}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    start = time.time()

    env, _ = make_env()
    env = Monitor(env)
    eval_env, _ = make_env()

    # ===== 观测归一化 (VecNormalize) =====
    # 必须先包装为 VecEnv (DummyVecEnv)，再应用 VecNormalize
    # norm_obs=True:  归一化观测（关键，解决 position/velocity 量级差异）
    # norm_reward=False: 不归一化 reward（保持 reward shaping 的原始尺度）
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=NORM_OBS, norm_reward=False)

    # 评估环境也归一化，但 training=False（不更新统计量，用训练环境的统计）
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecNormalize(eval_env, norm_obs=NORM_OBS, norm_reward=False, training=False)

    print(f"游戏: {GAME}")
    print(f"模型: {MODEL_FILE}")
    print(f"动作空间: n={env.action_space.n}, 观测空间: shape={env.observation_space.shape}")
    print(f"超参数: {PARAMS}")

    model = MaskablePPO(
        'MlpPolicy',
        env=env,
        device=PARAMS['device'],
        verbose=PARAMS['verbose'],
        learning_rate=PARAMS['learning_rate'],
        n_steps=PARAMS['n_steps'],
        batch_size=PARAMS['batch_size'],
        n_epochs=PARAMS['n_epochs'],
        gamma=PARAMS['gamma'],
        ent_coef=PARAMS['ent_coef'],
        clip_range=PARAMS['clip_range'],
        gae_lambda=PARAMS['gae_lambda'],
    )

    ckpt_cb = CheckpointCallback(
        save_freq=PARAMS['eval_freq'],
        save_path='./logs/',
        name_prefix=MODEL_FILE
    )
    eval_cb = EvalCallback(
        eval_env,
        eval_freq=PARAMS['eval_freq'],
        n_eval_episodes=PARAMS['n_eval_episodes'],
        save_prefix=MODEL_FILE,
        train_vec_normalize=env  # 传入训练环境的 VecNormalize，评估时同步统计
    )

    logger.info(f"开始训练 {GAME}, reward=reward_v01, total_timesteps={PARAMS['total_timesteps']}")
    model.learn(
        total_timesteps=PARAMS['total_timesteps'],
        callback=[ckpt_cb, eval_cb],
        use_masking=True
    )

    model.save(f"./logs/{MODEL_FILE}")
    # 保存归一化统计量（测试时必须加载，否则观测尺度不一致会导致预测错误）
    env.save(f"./logs/{MODEL_FILE}_vecnormalize.pkl")
    dur = int(time.time() - start)
    logger.info(f"训练完成, 模型: logs/{MODEL_FILE}, 时长: {dur//3600:02d}:{(dur%3600)//60:02d}:{dur%60:02d}")
    print(f"\n最佳模型 mean_reward: {eval_cb.best_mean_reward:.2f}")
    print(f"通关标准: mean_reward >= -110 (100 episodes)")

    env.close()
    eval_env.close()


# ============================ 测试 ============================
def play(use_best=True, episodes=10):
    """
    加载模型可视化运行

    参数:
        use_best: True 加载 _best 模型, False 加载最终模型
        episodes: 运行局数
    """
    suffix = "_best" if use_best else ""
    load_file = f"./logs/{MODEL_FILE}{suffix}"
    print(f"加载模型: {load_file}")

    env, tick = make_env()

    # ===== 观测归一化 (与训练时一致) =====
    # 测试时必须加载训练阶段保存的归一化统计量，否则观测尺度不匹配
    env = DummyVecEnv([lambda: env])
    env = VecNormalize.load(f"./logs/{MODEL_FILE}_vecnormalize.pkl", env)
    env.training = False   # 推理时不更新统计量
    env.norm_reward = False

    model = MaskablePPO.load(load_file, env=env)

    pygame.init()
    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption(f"MountainCar - {MODEL_FILE}{suffix}")
    clock = pygame.time.Clock()

    success = 0
    for ep in range(episodes):
        obs = env.reset()
        total_r = 0
        step = 0
        done = False

        while not done:
            clock.tick(tick)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    pygame.quit()
                    return

            action, _ = model.predict(obs, deterministic=True)
            # VecEnv 返回 batch 格式: obs(1,2), reward([x]), done([bool])
            obs, rewards, dones, infos = env.step(action)
            total_r += float(rewards[0])
            step += 1
            done = bool(dones[0])

            frame = env.render(mode='rgb_array')
            frame = frame.transpose((1, 0, 2))
            surface = pygame.surfarray.make_surface(frame)
            surface = pygame.transform.scale(surface, screen.get_size())
            screen.blit(surface, (0, 0))
            pygame.display.flip()

        pos = obs.reshape(-1)[0]
        ok = pos >= 0.5
        if ok:
            success += 1
        print(f"Episode {ep+1}: steps={step} reward={total_r:.2f} "
              f"pos={pos:.3f} {'✓ SUCCESS' if ok else '✗ FAIL'}")
        time.sleep(1)

    print(f"\n成功率: {success}/{episodes} ({success/episodes*100:.0f}%)")
    print(f"通关标准: 100 episodes 平均 reward >= -110")

    env.close()
    pygame.quit()


# ============================ 入口 ============================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f'{SCRIPT_NAME}: MountainCar-v0 MaskablePPO')
    parser.add_argument('--mode', choices=['train', 'play'], default='train',
                        help='train=训练, play=测试 (默认 train)')
    parser.add_argument('--final', action='store_true',
                        help='play 模式下加载最终模型而非 best 模型')
    parser.add_argument('--episodes', type=int, default=10,
                        help='play 模式运行局数 (默认 10)')
    args = parser.parse_args()

    if args.mode == 'train':
        train()
    else:
        play(use_best=not args.final, episodes=args.episodes)
