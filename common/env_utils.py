import gym
import numpy as np
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation, ResizeObservation
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecMonitor

# --- 1. 1:1 复刻 Viet Nguyen 的奖励逻辑 ---
class CustomReward(gym.Wrapper):
    def __init__(self, env, world=1, stage=1):
        super().__init__(env)
        self._current_score = 0
        self._current_x = 0
        # 记录当前的关卡信息，用于判定死锁坐标
        self.world = world
        self.stage = stage

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        
        # 提取信息
        score = info.get('score', 0)
        x_pos = info.get('x_pos', 0)
        y_pos = info.get('y_pos', 0)
        flag_get = info.get('flag_get', False)

        # --- A. 分数驱动奖励 ---
        # 原始代码: reward += (info["score"] - self.curr_score) / 40.
        reward += (score - self._current_score) / 40.0
        self._current_score = score

        # --- B. 终局强奖励 (Value Gap) ---
        # 原始代码: if done: if flag_get: +50 else: -50
        if done:
            if flag_get:
                reward += 50.0
            else:
                reward -= 50.0
        
        # --- C. 坐标死锁检测 (针对迷宫关卡) ---
        # 原始代码中有针对 7-4 和 4-4 的硬编码坐标检测
        # 如果进入死胡同，直接扣 50 分并强制结束
        if self.world == 7 and self.stage == 4:
            if (506 <= x_pos <= 832 and y_pos > 127) or \
               (832 < x_pos <= 1064 and y_pos < 80) or \
               (1113 < x_pos <= 1464 and y_pos < 191) or \
               (1579 < x_pos <= 1943 and y_pos < 191) or \
               (1946 < x_pos <= 1964 and y_pos >= 191) or \
               (1984 < x_pos <= 2060 and (y_pos >= 191 or y_pos < 127)) or \
               (2114 < x_pos < 2440 and y_pos < 191) or \
               (x_pos < self._current_x - 500): # 防止回退太多
                reward -= 50.0
                done = True
        
        if self.world == 4 and self.stage == 4:
            if (x_pos <= 1500 and y_pos < 127) or \
               (1588 <= x_pos < 2380 and y_pos >= 127):
                reward -= 50.0
                done = True

        self._current_x = x_pos
        
        # --- D. 缩放 ---
        # 原始代码: return reward / 10.
        return state, reward / 10.0, done, info

    def reset(self, **kwargs):
        self._current_score = 0
        self._current_x = 0
        return self.env.reset(**kwargs)

# --- 2. 解决画面闪烁的跳帧器 ---
class SkipFrameAndResize(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip
        # buffer 用于保存最近两帧，做 MaxPool 消除闪烁
        self.obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}
        for i in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            
            # 保存最后两帧
            if i == self.skip - 2: self.obs_buffer[0] = obs
            if i == self.skip - 1: self.obs_buffer[1] = obs
            
            if done:
                break
        
        # Max Pool: 取最后两帧的最大值，解决 NES 游戏精灵闪烁问题
        max_frame = self.obs_buffer.max(axis=0)
        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        self.obs_buffer.fill(0)
        obs = self.env.reset(**kwargs)
        self.obs_buffer[0] = obs
        return obs

# --- 3. 环境生成工厂 ---
def get_vec_env(world=1, stage=1, num_envs=8):
    """
    Viet Nguyen 的配置是：
    1. 专图专练 (SuperMarioBros-w-s-v0)
    2. GrayScale + Resize 84x84
    3. Stack 4 Frames
    """
    # 构造特定的关卡 ID
    env_id = f'SuperMarioBros-{world}-{stage}-v0'
    
    def make_env():
        env = gym_super_mario_bros.make(env_id)
        # 原始代码默认用 SIMPLE_MOVEMENT (7动作)，如果想更灵活可用 COMPLEX
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        
        # 应用核心奖励逻辑
        env = CustomReward(env, world, stage)
        
        # 图像处理: Skip -> Gray -> Resize
        # 注意顺序：先 Skip 聚合奖励，再转灰度
        env = SkipFrameAndResize(env, skip=4) 
        env = GrayScaleObservation(env, keep_dim=True)
        env = ResizeObservation(env, shape=84)
        
        return env

    # 多进程
    env = SubprocVecEnv([make_env for _ in range(num_envs)], start_method='spawn')
    
    # 叠加 4 帧 (这是 PPO 的标配，原代码 env.py 里也堆叠了 4 帧)
    env = VecFrameStack(env, n_stack=4, channels_order='last')
    
    # 监控 (记录日志)
    env = VecMonitor(env)
    
    return env