import os

from stable_baselines3.a2c import A2C
from stable_baselines3.common.utils import get_system_info
from stable_baselines3.ddpg import DDPG
from stable_baselines3.dqn import DQN
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.ppo import PPO
from stable_baselines3.sac import SAC
from stable_baselines3.td3 import TD3

from algorithm.phi_update import PhiUpdate
from algorithm.phi_PPO import PhiPPO

__all__ = [
    "A2C",
    "DDPG",
    "DQN",
    "PPO",
    "SAC",
    "TD3",
    "PhiUpdate",
    "PhiPPO",
    "HerReplayBuffer",
    "get_system_info",
]

ALGO = {
    "A2C": A2C,
    "DDPG": DDPG,
    "DQN": DQN,
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
    "PhiUpdate": PhiUpdate,
    "PhiPPO": PhiPPO
}