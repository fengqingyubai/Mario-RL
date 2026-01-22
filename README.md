

# 环境配置



```shell
conda create -n mario python=3.9 -y
# 激活环境
conda activate mario
python -m pip install pip==23.3.1
pip install -r requirements.txt
```

# 项目核心代码

`common\env_utils.py` 定义了奖励函数，马里奥动作等对环境的处理

`common\swanlab_callback.py` 用swanlab在网页监控服务器的训练损失

`train.py` 训练模型

`play.py`用训练好的模型开始玩游戏，把通关结果录制成视频

`checkpoints_general`子目录保存模型的checkpoints
