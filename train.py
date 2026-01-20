import os
import multiprocessing
import swanlab
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from common.env_utils import get_vec_env
from common.swanlab_callback import SwanLabCallback

if __name__ == '__main__':
    # --- 1. 设置要训练的关卡 ---
    # 原代码是针对单关卡设计的，这里我们练 1-1
    WORLD = 1
    STAGE = 1
    
    # 进程数 (原代码默认 8)
    # 如果你 CPU 强，可以设为 8；如果弱，SB3 建议设为核心数
    NUM_ENVS = 8 
    
    print(f"🚀 启动复刻版训练: World {WORLD}-{STAGE}, 进程数: {NUM_ENVS}")

    # --- 2. 初始化 SwanLab ---
    swanlab.init(
        project="SuperMario-RL", 
        experiment_name=f"PPO-VietNguyen-Rep-1-1",
        description="1:1复刻VietNguyen参数: LR=1e-4, Gamma=0.9, GAE=1.0, Score奖励",
        config={
            "algorithm": "PPO",
            "world": WORLD,
            "stage": STAGE,
            "num_envs": NUM_ENVS,
            # === 核心复刻参数 ===
            "learning_rate": 1e-4,      # 恒定，不衰减
            "n_steps": 512,             # 极短的采样长度 (更新频繁)
            "batch_size": 256,          # 4096 / 16 = 256
            "n_epochs": 10,             # 数据复习 10 遍
            "gamma": 0.9,               # 极度短视，只看眼前
            "gae_lambda": 1.0,          # 比较罕见的设置
            "clip_range": 0.2,
            "ent_coef": 0.01,           # 标准探索
            "max_grad_norm": 0.5,
            "vf_coef": 0.5,
        }
    )

    CHECKPOINT_DIR = f'./checkpoints_{WORLD}_{STAGE}/'
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # --- 3. 创建环境 ---
    env = get_vec_env(world=WORLD, stage=STAGE, num_envs=NUM_ENVS)

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
    # 原代码目标 500万步，这里我们设大一点，手动停止即可
    TOTAL_TIMESTEPS = 5000000 
    
    # 调整保存频率：原代码每 50 次 update 保存一次
    # 50 updates * 512 steps * 8 envs = 204,800 steps
    # 我们这里简化为每 20万步保存
    save_freq = 200000 // NUM_ENVS

    callbacks = CallbackList([
        CheckpointCallback(save_freq=save_freq, save_path=CHECKPOINT_DIR, name_prefix='mario_viet'),
        SwanLabCallback()
    ])

    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks) 
    except KeyboardInterrupt:
        print("检测到中断，正在保存模型...")
    finally:
        model.save(f"mario_viet_final_{WORLD}_{STAGE}")
        swanlab.finish()
        env.close()
        print("训练结束。")