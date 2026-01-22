import os
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles  # 👈 新增：用于提供静态文件访问
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from infer_core import record_gameplay

app = FastAPI(title="Mario AI Backend")

# --- 1. 解决跨域问题 (必须加!) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许任何来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 视频存储目录
OUTPUT_DIR = "videos_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. 【核心】挂载静态目录 ---
# 这样访问 http://localhost:8000/videos/xxx.mp4 就能直接看视频
app.mount("/videos", StaticFiles(directory=OUTPUT_DIR), name="videos")

class GameRequest(BaseModel):
    world: int
    stage: int
    deterministic: bool = True

@app.post("/generate_video")
def generate_video_endpoint(req: GameRequest):
    if not (1 <= req.world <= 8) or not (1 <= req.stage <= 4):
        raise HTTPException(status_code=400, detail="关卡范围错误")

    filename = f"replay_w{req.world}_s{req.stage}_{uuid.uuid4().hex[:8]}.mp4"
    file_path = os.path.join(OUTPUT_DIR, filename)

    try:
        print(f"🎥 处理请求: World {req.world}-{req.stage}")
        
        # 调用推理
        record_gameplay(
            world=req.world,
            stage=req.stage,
            deterministic=req.deterministic,
            output_path=file_path,
            checkpoint_root="./checkpoints"
        )
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=500, detail="视频生成失败")

        # --- 3. 【核心】返回 URL 而不是文件本身 ---
        # 假设你的服务器在本地，返回对应的访问链接
        video_url = f"http://localhost:8000/videos/{filename}"
        
        return {"status": "success", "video_url": video_url}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))