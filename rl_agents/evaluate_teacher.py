from contexts.offroad_env import OffRoadEnv
import numpy as np
import torch
from rl_agents.actor_critic_net import ActorCriticNet
from rl_agents.ppo import PPO
from rl_agents.buffer import RolloutBuffer
from cfg.jackal_config import JackalCfg
import time
import argparse
import glob
import wandb
import redis
import json
import os
from prettytable import PrettyTable
from contexts.terrain_context import TerrainContextSpace

def evaluate():
    parser = argparse.ArgumentParser(description="Train Teacher Model")

    parser.add_argument('--tif_name', type=str, default="Nullptr")
    parser.add_argument('--backbone', type=str, default="Nullptr")
    parser.add_argument('--wandb', type=bool, default=False)
    parser.add_argument('--project', type=str, default="Offroad Navigation")
    parser.add_argument('--method', type=str, default="NatureTerrain")
    parser.add_argument('--save', type=bool, default=False, help='whether to save the results')

    args = parser.parse_args()

    cfg = JackalCfg()
    cfg.state_space.use_noise = False
    cfg.state_space.use_globalmap = True
    
    env = OffRoadEnv(cfg, mode='rl', render=False, terrain_path=args.tif_name)
    env.max_goal_dis = 6.
    env.max_goal_ang = np.pi
    env.cfg.env.max_episode_length = env.max_goal_dis / 0.6 / env.cfg.sim.dt

    # Redis
    r = redis.Redis(host='localhost', port=6379, db=0)
    terrain_context_data = r.get('terrain_context')
    if terrain_context_data is None:
        raise ValueError("No terrain context found in Redis!")
    terrain_context_dict = json.loads(terrain_context_data)
    terrain_context = TerrainContextSpace.from_dict(terrain_context_dict)

    start_time = time.time()

    with torch.no_grad():
        actor_critic_net = ActorCriticNet(env.state_dim, env.action_dim, cfg.rl.net.action_noise_std, 
                                    cfg.rl.net.num_policy_network_layers,
                                    cfg.rl.net.num_value_network_layers, 
                                    cfg.rl.net.policy_network_hidden_dim,
                                    cfg.rl.net.value_network_hidden_dim,
                                    cfg.action_space.action_lower,
                                    cfg.action_space.action_upper)
        actor_critic_net.to(cfg.device)
        file_index = 0
        if args.backbone != 'Nullptr' and args.backbone[-3:] == '.pt':
            actor_critic_net.load_state_dict(torch.load(args.backbone))
        else:
            raise ValueError("No trained model.")
        actor_critic_net.eval()

        print('================= EVALUATE =================')
        
        iterations = 10 if args.save else 1
        for ite in range(iterations):
            num_successful_runs = torch.zeros(env.num_agents, device=cfg.device)
            total_rewards = torch.zeros(env.num_agents, device=cfg.device)

            state = env.reset()

            while True:
                action, _, _ = actor_critic_net.evaluate(state, stochastic=False)                    
                
                state, reward, done, timeout = env.step(action.detach().clone())
                
                num_successful_runs += env.goal_termination
                num_successful_runs[num_successful_runs != 0] = 1
                total_rewards += reward

                timeout_id = torch.count_nonzero(timeout)
                if timeout_id > 0:
                    log_data = {
                        'Time elapsed (mins)': (time.time() - start_time) / 60.,
                        'Total Time elapsed (mins)': (time.time() - terrain_context.start_time) / 60.,
                        'Episode length': cfg.env.max_episode_length,
                        'Model Name': args.backbone,
                        'Environment': args.tif_name
                    }
                    table = PrettyTable()
                    table.field_names = ["Metric", "Value"]
                    # Add log data to the table
                    for key, value in log_data.items():
                        table.add_row([key, value])
                    print(table)

                    total_success_reshaped = num_successful_runs.view(cfg.env.num_agents_per_terrain, 
                    cfg.context.environment.num_rows,
                    cfg.context.environment.num_cols)
                    envs_success_rate = (torch.sum(total_success_reshaped, dim=0) / cfg.env.num_agents_per_terrain).view(-1)
                    terrain_context.update(envs_success_rate.cpu().numpy(), file_index)

                    if args.save:
                        sum_reward = (torch.sum(total_rewards) / env.num_agents).item()
                        sum_success = (torch.sum(num_successful_runs) / env.num_agents).item()
                        terrain_context.add(args.backbone, ite, sum_reward, sum_success)
                    break
        r.set('terrain_context', json.dumps(terrain_context.to_dict()))

if __name__ == "__main__":
    evaluate()