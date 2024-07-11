import os
import yaml
import time
import torch
import numpy as np
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env

from algorithm import ALGO, PPO, PhiUpdate
from utils.function_bank import get_phi


if __name__ == '__main__':
    
    conf_path = "config/LunarLander-v2-phi.yaml"
    
    with open(conf_path, 'r', encoding="utf-8") as f:
        kwargs = yaml.load(f.read(), Loader=yaml.FullLoader)       
         
    torch.set_num_threads(kwargs["world"]["threads_num"])
    
    exp_dir = kwargs["world"]["exp_dir"]
    save_dir = os.path.join(exp_dir, str(int(time.time())))
    log_dir = os.path.join(save_dir, "log")
    ckpt_dir = os.path.join(save_dir, "ckpt")
    os.makedirs(save_dir)
    os.makedirs(log_dir)
    os.makedirs(ckpt_dir)
    cfg_path = os.path.join(save_dir, "config.yaml")
    with open(cfg_path, 'w') as f:
        yaml.dump(kwargs, f)         
    
    # env = gym.make(kwargs["env"]["env_name"])
    env = make_vec_env(**kwargs["env"])

    algo_id = kwargs["alg"]
    if algo_id == "PhiUpdate":
        phi = get_phi(**kwargs["phi"])
        model = PhiUpdate(**kwargs["algo"], phi=phi, env=env, verbose=1, tensorboard_log=log_dir)
    else:
        model = ALGO[algo_id](**kwargs["algo"], env=env, verbose=1, tensorboard_log=log_dir)
    
    model.learn(**kwargs["train"], progress_bar=True)
    model.save(os.path.join(ckpt_dir, "Final"))