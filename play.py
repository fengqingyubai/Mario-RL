import cv2
import os
import torch
import numpy as np
from stable_baselines3 import PPO
from common.env_utils import get_vec_env
from tqdm import tqdm

# --- 配置区 (必须与 train_paper.py 一致) ---
WORLD = 1
STAGE = 1
# 视频保存名称
OUTPUT_FILENAME = f"replay_viet_reproduced_{WORLD}-{STAGE}.mp4"
# 最大录制步数
MAX_STEPS = 5000 

def find_best_model(world, stage):
    """
    自动寻找 checkpoints 文件夹下步数最大的模型
    """
    # 1. 优先找最终保存的模型
    final_path = f"mario_viet_final_{world}_{stage}.zip"
    if os.path.exists(final_path):
        return final_path
    
    # 2. 没找到最终模型，去 checkpoint 文件夹找最新的
    ckpt_dir = f"./checkpoints_{world}_{stage}/"
    if not os.path.exists(ckpt_dir):
        return None
        
    import glob
    # 匹配 mario_viet_xxxxx_steps.zip
    files = glob.glob(os.path.join(ckpt_dir, "mario_viet_*_steps.zip"))
    if not files:
        return None
    
    # 按步数排序 (文件名 split 取数字)
    try:
        # x: .../mario_viet_200000_steps.zip
        # split('_')[-2] -> 200000
        best_file = max(files, key=lambda x: int(x.split('_')[-2]))
        return best_file
    except:
        return files[-1] # 兜底

if __name__ == '__main__':
    # --- 1. 寻找并加载模型 ---
    model_path = find_best_model(WORLD, STAGE)
    
    if not model_path:
        print(f"❌ 未找到 World {WORLD}-{STAGE} 的模型文件！")
        print(f"请检查是否运行过 train_paper.py，或手动修改 model_path。")
        exit()

    print(f"🔍 正在加载模型: {model_path}")
    
    # 显式指定 device (播放建议用 CPU，避免和训练抢显存，且足够快)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = PPO.load(model_path, device=device)
    except Exception as e:
        print(f"加载失败: {e}")
        exit()

    # --- 2. 创建环境 ---
    print(f"🎮 正在初始化环境 (SuperMarioBros-{WORLD}-{STAGE}-v0)...")
    
    # 【关键】这里 num_envs=1，但必须使用 get_vec_env
    # 这样才能保证 GrayScale, Resize(84x84), FrameStack(4) 等预处理完全一致
    env = get_vec_env(world=WORLD, stage=STAGE, num_envs=1)

    # --- 3. 视频录制准备 ---
    # 原始画面尺寸通常是 256x240
    width, height = 256, 240
    fps = 60.0 # 尝试 60帧录制，更流畅
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(OUTPUT_FILENAME, fourcc, fps, (width, height))

    print(f"🔴 开始录制... (进度条显示步数，满 {MAX_STEPS} 步自动停止)")

    # --- 4. 游戏循环 ---
    obs = env.reset()
    total_reward = 0
    current_episode_reward = 0
    episode_count = 1
    
    # 进度条
    pbar = tqdm(total=MAX_STEPS)

    try:
        for i in range(MAX_STEPS):
            # A. 获取高清彩色原图 (Render)
            # 这里的 render 拿到的不是 84x84 的灰度图，而是 gym 原始的 RGB 画面
            frame = env.render(mode='rgb_array')
            
            # VecEnv 的 render 有时返回 list
            if isinstance(frame, list): 
                frame = frame[0]
            
            # 写入视频
            if frame is not None:
                # 确保尺寸一致
                if frame.shape[0] != height or frame.shape[1] != width:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)
                # RGB -> BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame)

            # B. 模型预测
            # deterministic=True 是 PPO 验证的关键，去掉随机性，展示最强实力
            action, _ = model.predict(obs, deterministic=False)

            # C. 环境执行
            obs, reward, done, info = env.step(action)
            
            # 累加奖励 (注意 VecEnv 返回的是数组)
            current_episode_reward += reward[0]
            pbar.update(1)

            # D. 判断回合结束
            # 注意：在 env_utils 里我们写了 flag_get 的逻辑
            if done[0]:
                # 提取真实信息 (info 也是 list)
                inf = info[0]
                flag_status = "🚩 通关!" if inf.get('flag_get', False) else "💀 死亡"
                
                # 打印本局战报
                tqdm.write(f"局数: {episode_count} | 状态: {flag_status} | 原始得分: {inf.get('score', 0)} | 奖励分: {current_episode_reward:.2f}")
                
                episode_count += 1
                current_episode_reward = 0
                # VecEnv 会自动 reset，无需手动调用

    except KeyboardInterrupt:
        print("\n🛑 手动停止录制")
    finally:
        video_writer.release()
        env.close()
        pbar.close()
        print(f"\n✅ 录制完成！视频已保存至: {os.path.abspath(OUTPUT_FILENAME)}")