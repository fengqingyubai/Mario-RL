import swanlab
from stable_baselines3.common.callbacks import BaseCallback

class SwanLabCallback(BaseCallback):
    """
    一个自定义的 SB3 回调函数，用于将训练指标发送给 SwanLab
    """
    def __init__(self, verbose=0):
        super(SwanLabCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # SB3 在每一步结束时都会调用这个函数
        # self.locals['infos'] 包含了环境返回的信息
        
        for info in self.locals['infos']:
            # 'episode' 键只有在环境 reset (即 Mario 死掉或通关) 时才会出现
            # 里面包含了这一局的总奖励 (r) 和总时长 (l)
            if 'episode' in info:
                reward = info['episode']['r']
                length = info['episode']['l']
                
                # 发送到 SwanLab 云端
                swanlab.log({
                    "Env/Reward": reward,       # 这一局的总分
                    "Env/Episode_Length": length, # 这一局坚持了多久
                    "Train/Timesteps": self.num_timesteps # 当前总训练步数
                })
        
        return True