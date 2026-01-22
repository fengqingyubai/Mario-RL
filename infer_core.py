import cv2
import os
import torch
import glob
from stable_baselines3 import PPO
from common.env_utils import get_vec_env

# --- 关键配置 ---
# NES原始帧率是 60，我们跳了4帧
# 所以视频写入帧率应为 15，才能保证时间流逝速度正常
REAL_TIME_FPS = 15.0 

def find_best_model(checkpoint_root, world, stage):
    """查找最佳模型逻辑"""
    # 1. 优先找根目录最终模型
    final_name = os.path.join(checkpoint_root, f"final_model_level_{world}_{stage}.zip")
    # 如果根目录没有，尝试去具体的关卡文件夹找 (根据你的train.py逻辑调整)
    if os.path.exists(final_name):
        return final_name
    
    # 2. 找 checkpoints/level_x_x/ 下的
    ckpt_dir = os.path.join(checkpoint_root, f"level_{world}_{stage}")
    # 兼容 checkpoints_general 这种命名，根据你实际情况调整
    if not os.path.exists(ckpt_dir):
        # 尝试通用目录结构
        ckpt_dir = os.path.join(checkpoint_root, f"checkpoints_{world}_{stage}")
        
    if not os.path.exists(ckpt_dir):
        return None
    
    pattern = os.path.join(ckpt_dir, f"mario_{world}_{stage}_*_steps.zip")
    files = glob.glob(pattern)
    if not files:
        return None
        
    try:
        best_file = max(files, key=lambda x: int(x.split('_')[-2]))
        return best_file
    except:
        return files[-1]

def record_gameplay(world: int, stage: int, deterministic: bool, output_path: str, checkpoint_root="./checkpoints"):
    """
    执行推理并录制视频
    :param deterministic: True为确定性策略(展示最强实力), False为随机策略(更像人)
    """
    # 1. 寻找模型
    model_path = find_best_model(checkpoint_root, world, stage)
    if not model_path:
        raise FileNotFoundError(f"未找到 World {world}-{stage} 的模型文件")

    print(f"Loading model: {model_path}")

    # 2. 初始化环境 (强制 num_envs=1)
    # 注意：这里会复用你 env_utils.py 中的逻辑，包含 SkipFrame=4
    env = get_vec_env(world=world, stage=stage, num_envs=1)

    # 3. 加载模型 (使用 CPU 推理即可，避免显存冲突，且对于单进程推理足够快)
    model = PPO.load(model_path, device="cpu")

    # 4. 视频配置
    # 原始渲染尺寸通常是 256x240
    width, height = 256, 240
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # 【核心修改】这里使用 15 FPS，修复“视频太快”的问题
    video_writer = cv2.VideoWriter(output_path, fourcc, REAL_TIME_FPS, (width, height))

    obs = env.reset()
    
    # 限制最大步数，防止死循环 (约 3分钟游戏时间)
    # 15 fps * 180s = 2700 frames
    max_frames = 3000 
    
    try:
        for _ in range(max_frames):
            # 获取画面
            frame = env.render(mode='rgb_array')
            if isinstance(frame, list): frame = frame[0]

            if frame is not None:
                # 调整尺寸兼容性
                if frame.shape[0] != height or frame.shape[1] != width:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)
                # RGB 转 BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame)

            # 预测
            action, _ = model.predict(obs, deterministic=deterministic)
            
            # 执行
            obs, reward, done, info = env.step(action)
            
            if done[0]:
                break
    finally:
        # 清理资源
        video_writer.release()
        env.close()
        del model # 显式释放模型内存
        
    return True