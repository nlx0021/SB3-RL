import os
import yaml
import torch
import numpy as np
import gymnasium as gym

from algorithm import ALGO
from utils.function_bank import get_phi

 
if __name__ == '__main__':
    
    conf_path = "exp\\LunarLander-v2\\Phi-power-5-3-baseline\\config.yaml"
    with open(conf_path, 'r', encoding="utf-8") as f:
        kwargs = yaml.load(f.read(), Loader=yaml.FullLoader)        
    
    torch.set_num_threads(kwargs["world"]["threads_num"])
    env = gym.make(kwargs["env"]["env_id"], render_mode="human") 
    
    ckpt_path = "exp\\LunarLander-v2\\Phi-power-5-3-baseline\\ckpt\\Final.zip"
    if kwargs["alg"] in ["PhiUpdate", "PhiPPO"]:
        phi = get_phi(**kwargs["phi"]) 
        model = ALGO[kwargs["alg"]].load(ckpt_path, env=env, phi=phi)
    else:
        model = ALGO[kwargs["alg"]].load(ckpt_path, env=env)
    
    obs, info = env.reset()
    total_reward = 0
    game_ct = 0
    while game_ct < 100:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            observation, info = env.reset()    
            print("Over: %d. The reward is %f" % (game_ct, total_reward))
            game_ct += 1
            total_reward = 0        
        # vec_env.render()
        # VecEnv resets automatically
        # if done:
        #   obs = env.reset()

    env.close()    