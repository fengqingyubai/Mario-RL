import os
import uuid
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from infer_core import record_gameplay

app = FastAPI(title="Mario AI Backend")

# 视频暂存目录
OUTPUT_DIR = "videos_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 定义请求参数结构
class GameRequest(BaseModel):
    world: int
    stage: int
    deterministic: bool = True # 默认使用确定性最强策略

@app.get("/")
def health_check():
    return {"status": "ok", "msg": "Mario AI Service is running"}

@app.post("/generate_video")
def generate_video_endpoint(req: GameRequest):
    """
    接收关卡和策略，返回生成的视频文件
    """
    # 简单的参数校验
    if not (1 <= req.world <= 8) or not (1 <= req.stage <= 4):
        raise HTTPException(status_code=400, detail="关卡范围错误，World:1-8, Stage:1-4")

    # 生成唯一的视频文件名，避免冲突
    filename = f"replay_w{req.world}_s{req.stage}_{uuid.uuid4().hex[:8]}.mp4"
    file_path = os.path.join(OUTPUT_DIR, filename)

    try:
        print(f"收到请求: World {req.world}-{req.stage}, Det={req.deterministic}")
        
        # 调用核心推理逻辑
        record_gameplay(
            world=req.world,
            stage=req.stage,
            deterministic=req.deterministic,
            output_path=file_path,
            checkpoint_root="./checkpoints" # 你的模型根目录
        )
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=500, detail="视频生成失败")

        # 返回视频文件
        # media_type="video/mp4" 让浏览器可以直接播放
        return FileResponse(file_path, media_type="video/mp4", filename=filename)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="服务器内部错误")

# 清理任务（可选）：定期清理旧视频，或者在返回后删除（需要BackgroundTasks）