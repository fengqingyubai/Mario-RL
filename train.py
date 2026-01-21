import os
import multiprocessing
import swanlab
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from common.env_utils import get_vec_env
from common.swanlab_callback import SwanLabCallback

# --- 线性学习率调度器 (备用，当前配置使用固定 LR) ---
def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

if __name__ == '__main__':
    # --- 1. 配置 ---
    # 根据 CPU 核心数自动调整进程数，或者手动指定 (如 8)
    cpu_count = multiprocessing.cpu_count()
    NUM_ENVS = min(16, max(4, int(cpu_count * 0.8))) 
    
    print(f"🚀 启动全关卡通用训练 (SuperMarioBros-v0), 进程数: {NUM_ENVS}")

    # --- 2. 初始化 SwanLab ---
    swanlab.init(
        project="SuperMario-RL", 
        experiment_name="PPO-Mario-AllLevels-20M",
        description="全关卡训练 PPO (LR=1e-4, Gamma=0.9, GAE=1.0) - 目标 2000万步",
        config={
            "algorithm": "PPO",
            "env": "SuperMarioBros-v0", # 全关卡
            "num_envs": NUM_ENVS,
            # === 沿用之前的成功参数 ===
            "learning_rate": 1e-4,      # 固定 1e-4，稳健
            "n_steps": 512,             # 短采样
            "batch_size": 256,          # 4096 / 16
            "n_epochs": 10,             
            "gamma": 0.9,               # 短视策略，适合动作游戏
            "gae_lambda": 1.0,          
            "clip_range": 0.2,
            "ent_coef": 0.01,           
            "max_grad_norm": 0.5,
            "vf_coef": 0.5,
        }
    )

    # 检查点保存目录
    CHECKPOINT_DIR = './checkpoints_general/'
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # --- 3. 创建环境 ---
    # 不再传入 world/stage，默认加载全关卡环境
    env = get_vec_env(num_envs=NUM_ENVS)

    # --- 4. 构建模型 ---
    model = PPO(
        'CnnPolicy', 
        env, 
        verbose=1, 
        
        # 参数映射
        learning_rate=swanlab.config["learning_rate"], 
        n_steps=swanlab.config["n_steps"],     
        batch_size=swanlab.config["batch_size"], 
        n_epochs=swanlab.config["n_epochs"],
        gamma=swanlab.config["gamma"],
        gae_lambda=swanlab.config["gae_lambda"],
        clip_range=swanlab.config["clip_range"],
        ent_coef=swanlab.config["ent_coef"],
        max_grad_norm=swanlab.config["max_grad_norm"],
        
        device="cuda", 
        tensorboard_log=None 
    )

    # --- 5. 开始训练 ---
    # 目标：2000万步 (20M)
    TOTAL_TIMESTEPS = 20000000 
    
    # 保存频率：每 50万步保存一次
    # 计算方式：500,000 / 进程数
    save_freq = max(1, 500000 // NUM_ENVS)

    callbacks = CallbackList([
        CheckpointCallback(save_freq=save_freq, save_path=CHECKPOINT_DIR, name_prefix='mario_general'),
        SwanLabCallback()
    ])

    try:
        print(f"开始训练! 目标步数: {TOTAL_TIMESTEPS}")
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks) 
    except KeyboardInterrupt:
        print("检测到中断，正在保存模型...")
    finally:
        # 保存最终模型
        model.save("mario_general_final_20M")
        swanlab.finish()
        env.close()
        print("训练结束，资源已释放。")