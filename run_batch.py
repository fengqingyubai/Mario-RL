import subprocess
import time
import sys

# Python 解释器路径
PYTHON_EXE = sys.executable 

# --- 配置区 ---
# 在这里填入你想使用的显卡 ID 列表
# 如果是卡1和卡2，通常 ID 是 [1, 2]
# 如果你是指前两张卡，ID 通常是 [0, 1]，请根据 nvidia-smi 确认
USE_GPUS = [1, 2] 

def run_world(world_id, num_envs=4):
    """
    启动一个 World 下的所有 Stage (1-4)，并自动分配到多张显卡
    """
    processes = []
    print(f"⚡ [World {world_id}] 启动中... 任务将分配到 GPUs: {USE_GPUS}")
    
    for stage in range(1, 5):
        # 简单的轮询分配算法 (Round-Robin)
        # stage 1 -> GPU A
        # stage 2 -> GPU B
        # stage 3 -> GPU A
        # stage 4 -> GPU B
        gpu_index = (stage - 1) % len(USE_GPUS)
        target_gpu = USE_GPUS[gpu_index]

        cmd = [
            PYTHON_EXE, "train.py",
            "--world", str(world_id),
            "--stage", str(stage),
            "--num_envs", str(num_envs), 
            "--gpu_id", str(target_gpu) # 指定这一关用哪张卡
        ]
        
        # 启动训练进程
        p = subprocess.Popen(cmd)
        processes.append(p)
        
        print(f"  -> 关卡 {world_id}-{stage} 已启动 (PID: {p.pid}) 运行于 GPU: {target_gpu}")
        
        # 稍微错峰启动，避免 CPU 瞬间负载过高
        time.sleep(3) 

    print(f"⏳ [World {world_id}] 正在训练，请耐心等待所有关卡完成...")
    
    # 阻塞等待这 4 个任务全部结束
    for p in processes:
        p.wait()
    
    print(f"✅ [World {world_id}] 所有关卡训练完毕！")

if __name__ == '__main__':
    # 依次训练 World 1 到 World 8
    for w in range(1, 9):
        # 注意：这里 num_envs=4
        # 虽然我们用了两张显卡，但 CPU 依然要同时跑 4个训练任务 x 4个环境 = 16 个 Python 进程
        # 请确保 CPU 核心数够用
        run_world(world_id=w, num_envs=4)
        
        print("------------------------------------------------")
        print(f"休息 10 秒，准备进入 World {w + 1}...")
        time.sleep(10)