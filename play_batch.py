import cv2
import os
import torch
import glob
import argparse
import numpy as np
from stable_baselines3 import PPO
from common.env_utils import get_vec_env
from tqdm import tqdm

# --- 🎯 配置区域 ---
# 录制范围设置
START_WORLD = 1
START_STAGE = 1
END_WORLD = 3
END_STAGE = 4

# 每个关卡录制的最大步数 (防止死循环)
MAX_STEPS_PER_LEVEL = 4000 

# 视频保存目录
VIDEO_DIR = "videos_batch"

# 模型查找路径 (根据你的 train.py 设置)
# 1. 优先找根目录下的 final_model_level_X_X.zip
# 2. 其次找 checkpoints/level_X_X/ 下步数最大的检查点
CHECKPOINT_ROOT = "./checkpoints"

def find_best_model(world, stage):
    """
    智能查找对应关卡的最佳模型权重
    """
    # 策略 A: 找根目录的最终模型 (final_model_level_1_1.zip)
    final_name = f"final_model_level_{world}_{stage}.zip"
    if os.path.exists(final_name):
        return final_name
    
    # 策略 B: 找 Checkpoint 目录下的最新权重
    ckpt_dir = os.path.join(CHECKPOINT_ROOT, f"level_{world}_{stage}")
    if not os.path.exists(ckpt_dir):
        return None
    
    # 匹配 mario_1_1_500000_steps.zip 格式
    pattern = os.path.join(ckpt_dir, f"mario_{world}_{stage}_*_steps.zip")
    files = glob.glob(pattern)
    
    if not files:
        return None
    
    # 按步数排序，取最大的
    try:
        # 文件名示例: .../mario_1_1_500000_steps.zip
        # split('_')[-2] 拿到 500000
        best_file = max(files, key=lambda x: int(x.split('_')[-2]))
        return best_file
    except:
        return files[-1] # 兜底策略

def record_level(world, stage):
    """
    录制单个关卡的视频
    """
    # 1. 寻找模型
    model_path = find_best_model(world, stage)
    if not model_path:
        print(f"⚠️ [跳过] 未找到 World {world}-{stage} 的模型文件")
        return
    
    print(f"\n🎬 正在准备录制 World {world}-{stage} ...")
    print(f"   📂 加载模型: {model_path}")

    # 2. 初始化环境 (必须与训练时一致)
    # 强制 num_envs=1 用于录制
    try:
        env = get_vec_env(world=world, stage=stage, num_envs=1)
    except Exception as e:
        print(f"❌ 环境创建失败: {e}")
        return

    # 3. 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = PPO.load(model_path, device=device)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        env.close()
        return

    # 4. 视频写入器配置
    os.makedirs(VIDEO_DIR, exist_ok=True)
    video_name = f"replay_world_{world}_{stage}.mp4"
    video_path = os.path.join(VIDEO_DIR, video_name)
    
    # NES 标准分辨率通常渲染出来是 256x240
    width, height = 256, 240
    fps = 60.0 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # 5. 游戏循环
    obs = env.reset()
    total_reward = 0
    done_once = False
    
    # 进度条
    pbar = tqdm(total=MAX_STEPS_PER_LEVEL, desc=f"Recording {world}-{stage}")
    
    for _ in range(MAX_STEPS_PER_LEVEL):
        # A. 获取高清画面
        frame = env.render(mode='rgb_array')
        if isinstance(frame, list): frame = frame[0]

        if frame is not None:
            # 调整尺寸确保兼容
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)

        # B. 预测 (deterministic=True 展示最强实力)
        action, _ = model.predict(obs, deterministic=False)
        
        # C. 执行
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
        pbar.update(1)

        # D. 结束判定
        if done[0]:
            inf = info[0]
            flag = inf.get('flag_get', False)
            status = "🚩 通关" if flag else "💀 死亡"
            tqdm.write(f"   📊 结果: {status} | 得分: {inf.get('score', 0)}")
            done_once = True
            break  # 一局定胜负，死掉或者通关就停止录制这一关

    pbar.close()
    video_writer.release()
    env.close()
    print(f"   ✅ 视频已保存: {video_path}")

if __name__ == '__main__':
    print(f"🚀 开始批量录制: World {START_WORLD}-{START_STAGE} 到 {END_WORLD}-{END_STAGE}")
    
    # 遍历所有指定的关卡
    # 简单的双层循环，你可以根据需要修改逻辑
    for w in range(START_WORLD, END_WORLD + 1):
        for s in range(1, 5):
            # 处理起始和结束的边界条件
            if w == START_WORLD and s < START_STAGE: continue
            if w == END_WORLD and s > END_STAGE: break
            
            record_level(w, s)
            
    print("\n🎉 所有录制任务完成！请查看 videos_batch 文件夹。")