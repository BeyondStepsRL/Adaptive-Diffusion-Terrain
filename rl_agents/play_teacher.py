from contexts.offroad_env import OffRoadEnv
import numpy as np
import torch
from rl_agents.actor_critic_net import ActorCriticNet
from rl_agents.ppo import PPO
from rl_agents.buffer import RolloutBuffer
from cfg.jackal_config import JackalCfg
import time
import argparse
import json
import os

def play():
    parser = argparse.ArgumentParser(description="Play Teacher Model")

    parser.add_argument('--tif_name', type=str, default="assets/elevation/train/test_map1.txt")
    parser.add_argument('--backbone', type=str, default="Nullptr")
    parser.add_argument('--num_cols', type=int, default=1)
    parser.add_argument('--num_rows', type=int, default=1)
    parser.add_argument('--num_agents_per_terrain', type=int, default=20)

    args = parser.parse_args()

    cfg = JackalCfg()
    cfg.state_space.use_noise = False
    cfg.state_space.use_globalmap = True
    cfg.state_space.use_localmap = False

    cfg.context.environment.num_cols = args.num_cols
    cfg.context.environment.num_rows = args.num_rows
    cfg.env.num_agents_per_terrain = args.num_agents_per_terrain
    
    env = OffRoadEnv(cfg, mode='rl', render=True, terrain_path=args.tif_name)
    env.max_goal_dis = 6.
    env.max_goal_ang = np.pi
    env.cfg.env.max_episode_length = env.max_goal_dis / 0.6 / env.cfg.sim.dt

    with torch.no_grad():
        actor_critic_net = ActorCriticNet(env.state_dim, env.action_dim, cfg.rl.net.action_noise_std, 
                                    cfg.rl.net.num_policy_network_layers,
                                    cfg.rl.net.num_value_network_layers, 
                                    cfg.rl.net.policy_network_hidden_dim,
                                    cfg.rl.net.value_network_hidden_dim,
                                    cfg.action_space.action_lower,
                                    cfg.action_space.action_upper)
        actor_critic_net.to(cfg.device)

        if args.backbone != 'Nullptr' and args.backbone[-3:] == '.pt':
            actor_critic_net.load_state_dict(torch.load(args.backbone))
          
        actor_critic_net.eval()
        
        for itr in range(10):

            state = env.reset(None, reset_pos)
            
            while True:
                action, _, _ = actor_critic_net.evaluate(state, stochastic=False)
                # action[:, 0] = 0.5
                # action[:, 1] = 0
                
                env.step(action.detach().clone())

if __name__ == "__main__":
    play()
