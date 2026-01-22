import os
import argparse
import multiprocessing
import swanlab
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from common.env_utils import get_vec_env
from common.swanlab_callback import SwanLabCallback

# 接收命令行参数 (这是分关卡训练的关键)
def parse_args():
    parser = argparse.ArgumentParser(description="Train Mario RL per level")
    parser.add_argument("--world", type=int, required=True, help="World ID (1-8)")
    parser.add_argument("--stage", type=int, required=True, help="Stage ID (1-4)")
    parser.add_argument("--gpu_id", type=int, default=0, help="CUDA Device ID")
    parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel environments")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    WORLD = args.world
    STAGE = args.stage
    
    # 设置可见显卡
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    print(f"🚀 启动专用训练: World {WORLD}-{STAGE} | Envs: {args.num_envs}")

    # --- SwanLab 初始化 ---
    swanlab.init(
        project="SuperMario-RL-AllLevels", 
        experiment_name=f"Level-{WORLD}-{STAGE}", 
        description=f"专享模型训练: {WORLD}-{STAGE}",
        config={
            "algorithm": "PPO",
            "world": WORLD,
            "stage": STAGE,
            "num_envs": args.num_envs,
            # === Viet Nguyen 复刻版参数 ===
            "learning_rate": 1e-4,
            "n_steps": 512,
            "batch_size": 256,
            "n_epochs": 10,
            "gamma": 0.9,
            "gae_lambda": 1.0,
            "ent_coef": 0.01,
            "clip_range": 0.2,       
            "max_grad_norm": 0.5,    
            "vf_coef": 0.5,          
            "total_timesteps": 1500000 
        }
    )

    # 独立的保存目录
    CHECKPOINT_DIR = f'./checkpoints/level_{WORLD}_{STAGE}/'
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # --- 创建专用环境 (修复点：传入 world 和 stage) ---
    env = get_vec_env(world=WORLD, stage=STAGE, num_envs=args.num_envs)

    # --- 构建模型 ---
    model = PPO(
        'CnnPolicy', 
        env, 
        verbose=1, 
        # 读取配置参数
        learning_rate=swanlab.config["learning_rate"], 
        n_steps=swanlab.config["n_steps"],     
        batch_size=swanlab.config["batch_size"], 
        n_epochs=swanlab.config["n_epochs"],
        gamma=swanlab.config["gamma"],
        gae_lambda=swanlab.config["gae_lambda"],
        ent_coef=swanlab.config["ent_coef"],
        clip_range=swanlab.config["clip_range"],
        max_grad_norm=swanlab.config["max_grad_norm"],
        vf_coef=swanlab.config["vf_coef"],
        
        device="cuda", 
        tensorboard_log=None 
    )

    # --- 训练参数 ---
    TOTAL_TIMESTEPS = swanlab.config["total_timesteps"]
    
    # 每 50万步保存一次
    save_freq = max(1, 500000 // args.num_envs)

    callbacks = CallbackList([
        CheckpointCallback(save_freq=save_freq, save_path=CHECKPOINT_DIR, name_prefix=f'mario_{WORLD}_{STAGE}'),
        SwanLabCallback()
    ])

    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks) 
    except KeyboardInterrupt:
        print(f"Level {WORLD}-{STAGE} 中断，保存中...")
    finally:
        model.save(f"final_model_level_{WORLD}_{STAGE}")
        swanlab.finish()
        env.close()
        print(f"Level {WORLD}-{STAGE} 训练结束。")