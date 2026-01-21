import gym
import numpy as np
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation, ResizeObservation
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecMonitor

# --- 1. 通用奖励与防卡死逻辑 ---
class CustomReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._current_score = 0
        self._current_x = 0
        # 这里不再需要初始化传入 world/stage，改为动态获取

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        
        # 提取信息
        score = info.get('score', 0)
        x_pos = info.get('x_pos', 0)
        y_pos = info.get('y_pos', 0)
        flag_get = info.get('flag_get', False)
        
        # 动态获取当前关卡信息 (用于判断特殊关卡的死锁)
        # gym-super-mario-bros 的 info 中通常包含 'world' 和 'stage'
        current_world = info.get('world', 1)
        current_stage = info.get('stage', 1)

        # --- A. 分数驱动奖励 ---
        # 鼓励 杀怪、吃金币、打砖块
        reward += (score - self._current_score) / 40.0
        self._current_score = score

        # --- B. 终局强奖励 (Value Gap) ---
        # 通关给大奖，中途死亡给惩罚
        if done:
            if flag_get:
                reward += 50.0
            else:
                reward -= 50.0
        
        # --- C. 坐标死锁检测 (针对特定迷宫关卡) ---
        # 这些坐标是针对 7-4 和 4-4 的死循环路径检测
        if current_world == 7 and current_stage == 4:
            if (506 <= x_pos <= 832 and y_pos > 127) or \
               (832 < x_pos <= 1064 and y_pos < 80) or \
               (1113 < x_pos <= 1464 and y_pos < 191) or \
               (1579 < x_pos <= 1943 and y_pos < 191) or \
               (1946 < x_pos <= 1964 and y_pos >= 191) or \
               (1984 < x_pos <= 2060 and (y_pos >= 191 or y_pos < 127)) or \
               (2114 < x_pos < 2440 and y_pos < 191) or \
               (x_pos < self._current_x - 500): 
                reward -= 50.0
                done = True
        
        if current_world == 4 and current_stage == 4:
            if (x_pos <= 1500 and y_pos < 127) or \
               (1588 <= x_pos < 2380 and y_pos >= 127):
                reward -= 50.0
                done = True

        self._current_x = x_pos
        
        # --- D. 缩放 ---
        # 保持数值在较小范围，利于 PPO 收敛
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
            
            if i == self.skip - 2: self.obs_buffer[0] = obs
            if i == self.skip - 1: self.obs_buffer[1] = obs
            
            if done:
                break
        
        max_frame = self.obs_buffer.max(axis=0)
        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        self.obs_buffer.fill(0)
        obs = self.env.reset(**kwargs)
        self.obs_buffer[0] = obs
        return obs

# --- 3. 环境生成工厂 ---
def get_vec_env(num_envs=8):
    """
    配置：
    1. SuperMarioBros-v0 (包含所有关卡，通关自动下一关)
    2. GrayScale + Resize 84x84
    3. Stack 4 Frames
    """
    # 使用包含所有关卡的标准环境 ID
    env_id = 'SuperMarioBros-v0'
    
    def make_env():
        env = gym_super_mario_bros.make(env_id)
        # 依然使用 SIMPLE_MOVEMENT (7动作) 以加快收敛
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        
        # 应用通用奖励逻辑
        env = CustomReward(env)
        
        # 图像处理
        env = SkipFrameAndResize(env, skip=4) 
        env = GrayScaleObservation(env, keep_dim=True)
        env = ResizeObservation(env, shape=84)
        
        return env

    # 多进程启动
    env = SubprocVecEnv([make_env for _ in range(num_envs)], start_method='spawn')
    
    # 叠加 4 帧
    env = VecFrameStack(env, n_stack=4, channels_order='last')
    
    # 监控
    env = VecMonitor(env)
    
    return env