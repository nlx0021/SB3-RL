import os
import yaml
import torch
import numpy as np
import gymnasium as gym

from algorithm import ALGO

 
if __name__ == '__main__':
    
    conf_path = "config/LunarLander-v2-PPO.yaml"
    with open(conf_path, 'r', encoding="utf-8") as f:
        kwargs = yaml.load(f.read(), Loader=yaml.FullLoader)        
    
    torch.set_num_threads(kwargs["world"]["threads_num"])
    env = gym.make(kwargs["env"]["env_name"]) 
    
    ckpt_path = "******"
    model = ALGO[kwargs["alg"]].load(ckpt_path, env=env)
    
    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render()
        # VecEnv resets automatically
        # if done:
        #   obs = env.reset()

    env.close()    