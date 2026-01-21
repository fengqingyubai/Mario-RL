import cv2
import os
import torch
import numpy as np
from stable_baselines3 import PPO
from common.env_utils import get_vec_env
from tqdm import tqdm

# --- Configuration Area ---
# Video output filename
OUTPUT_FILENAME = "replay_general_20M.mp4"
# Maximum recording steps (increase this to see level transitions)
MAX_STEPS = 5000 

def find_best_model():
    """
    Automatically find the best model in the general checkpoints folder.
    """
    # 1. Priority: Look for the final saved model from the 20M run
    final_path = "mario_general_final_20M.zip"
    if os.path.exists(final_path):
        return final_path
    
    # 2. Fallback: Look for the latest checkpoint in the folder
    ckpt_dir = "./checkpoints_general/"
    if not os.path.exists(ckpt_dir):
        return None
        
    import glob
    # Match filenames like mario_general_500000_steps.zip
    files = glob.glob(os.path.join(ckpt_dir, "mario_general_*_steps.zip"))
    if not files:
        return None
    
    # Sort by step count
    try:
        best_file = max(files, key=lambda x: int(x.split('_')[-2]))
        return best_file
    except:
        return files[-1]

if __name__ == '__main__':
    # --- 1. Find and Load Model ---
    model_path = find_best_model()
    
    if not model_path:
        print(f"❌ No general model found!")
        print(f"Please check if 'train_general.py' has been run or manually set model_path.")
        exit()

    print(f"🔍 Loading model: {model_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = PPO.load(model_path, device=device)
    except Exception as e:
        print(f"Load failed: {e}")
        exit()

    # --- 2. Create Environment ---
    print(f"🎮 Initializing Environment (SuperMarioBros-v0 - All Levels)...")
    
    # [CRITICAL] num_envs=1, but must use get_vec_env to match training preprocessing
    # This function inside env_utils defaults to 'SuperMarioBros-v0' now
    env = get_vec_env(num_envs=1)

    # --- 3. Video Recording Setup ---
    width, height = 256, 240
    fps = 60.0 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(OUTPUT_FILENAME, fourcc, fps, (width, height))

    print(f"🔴 Start Recording... (Max Steps: {MAX_STEPS})")

    # --- 4. Game Loop ---
    obs = env.reset()
    total_reward = 0
    current_episode_reward = 0
    episode_count = 1
    
    # Progress bar
    pbar = tqdm(total=MAX_STEPS)

    try:
        for i in range(MAX_STEPS):
            # A. Get High-Res Color Frame
            frame = env.render(mode='rgb_array')
            
            if isinstance(frame, list): 
                frame = frame[0]
            
            if frame is not None:
                if frame.shape[0] != height or frame.shape[1] != width:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame)

            # B. Model Prediction
            # deterministic=True: Best performance
            # deterministic=False: More variety (good if AI gets stuck)
            action, _ = model.predict(obs, deterministic=False)

            # C. Step
            obs, reward, done, info = env.step(action)
            
            current_episode_reward += reward[0]
            pbar.update(1)

            # D. Handle Episode End
            if done[0]:
                inf = info[0]
                # In general mode, 'flag_get' might just mean level clear, not game over
                flag_status = "🚩 Level Clear!" if inf.get('flag_get', False) else "💀 Died"
                
                # Print stats
                # Note: In general mode, world/stage changes dynamically
                current_world = inf.get('world', '?')
                current_stage = inf.get('stage', '?')
                
                tqdm.write(f"Ep: {episode_count} | Level: {current_world}-{current_stage} | Status: {flag_status} | Score: {inf.get('score', 0)}")
                
                episode_count += 1
                current_episode_reward = 0

    except KeyboardInterrupt:
        print("\n🛑 Recording stopped manually")
    finally:
        video_writer.release()
        env.close()
        pbar.close()
        print(f"\n✅ Recording Complete! Saved to: {os.path.abspath(OUTPUT_FILENAME)}")