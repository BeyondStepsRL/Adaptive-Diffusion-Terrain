from contexts.offroad_env import OffRoadEnv
from contexts.terrain_context import TerrainContextSpace
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
import sys
from prettytable import PrettyTable

def train():
    parser = argparse.ArgumentParser(description="Train Teacher Model")

    parser.add_argument('--tif_name', type=str, default="Nullptr", help='tif name for terrains')
    parser.add_argument('--backbone', type=str, default="Nullptr", help='model to load')
    parser.add_argument('--project', type=str, default="Offroad Navigation")
    parser.add_argument('--method', type=str, default="NatureTerrain")

    parser.add_argument('--wandb', type=bool, default=False, help='https://wandb.ai')
    parser.add_argument('--wandb_entity', type=str, default="Nullptr")
    parser.add_argument('--wandb_id', type=str, default="Nullptr")

    args = parser.parse_args()

    if args.wandb:
        wandb.init(project=args.project, entity=args.wandb_entity, name=args.method, id=args.wandb_id, resume="allow")

    cfg = JackalCfg()
    cfg.state_space.use_noise = False
    cfg.state_space.use_globalmap = True

    # cfg.context.environment.num_cols = 1
    # cfg.context.environment.num_rows = 1
    # cfg.env.num_agents_per_terrain = 2
    
    # Redis
    r = redis.Redis(host='localhost', port=6379, db=0)
    terrain_context_data = r.get('terrain_context')
    if terrain_context_data is None:
        raise ValueError("No terrain context found in Redis!")
    terrain_context_dict = json.loads(terrain_context_data)
    terrain_context = TerrainContextSpace.from_dict(terrain_context_dict)

    env = OffRoadEnv(cfg, mode='rl', render=False, terrain_path=args.tif_name)
    env.cfg.env.max_episode_length = env.max_goal_dis / 0.6 / env.cfg.sim.dt
    
    state_dim = env.state_dim
    action_dim = env.action_dim
    num_learning_steps = cfg.rl.num_learning_steps
    # number of samples trained is equal to current_train_step * num_envs
    total_num_samples = int(cfg.rl.buffer.num_transitions_per_env * num_learning_steps * env.num_agents)

    buffer = RolloutBuffer(env.num_agents,
                           cfg.rl.buffer.num_transitions_per_env, 
                           state_dim, 
                           action_dim,
                           gamma=0.99)
    
    actor_critic_net = ActorCriticNet(state_dim, action_dim, cfg.rl.net.action_noise_std, 
                                      cfg.rl.net.num_policy_network_layers,
                                      cfg.rl.net.num_value_network_layers, 
                                      cfg.rl.net.policy_network_hidden_dim,
                                      cfg.rl.net.value_network_hidden_dim,
                                      cfg.action_space.action_lower,
                                      cfg.action_space.action_upper)
    actor_critic_net.to(cfg.device)
    
    if args.backbone != 'Nullptr' and args.backbone[-3:] == '.pt':
        actor_critic_net.load_state_dict(torch.load(args.backbone))

    ppo = PPO(buffer, actor_critic_net, cfg.rl.ppo.lr, cfg.rl.ppo.ppo_epoch, 
            cfg.rl.ppo.mini_batch_size, cfg.rl.ppo.clip_param, cfg.rl.ppo.value_loss_coeff, 
            entropy_coeff=cfg.rl.ppo.entropy_coeff, tensorboard_writer=None, 
            clip_grad_norm=cfg.rl.ppo.clip_grad_norm)

    # logged statistics
    envs_success_rate = None
    num_samples_visited = 0
    num_ppo_updates = 0
    # sum of rewards per environment
    total_reward = torch.zeros(env.num_agents, device=cfg.device)
    # sum of returns per environment of all the episodes
    total_episodic_returns = torch.zeros(env.num_agents, device=cfg.device) 
    # number of episodes, used to compute the averaged statistics
    num_episodes_per_env = torch.zeros(env.num_agents, device=cfg.device)
    num_unsuccessful_runs = torch.zeros(env.num_agents, device=cfg.device)
    num_successful_runs = torch.zeros(env.num_agents, device=cfg.device)
    success_rates = torch.zeros(env.num_agents, device=cfg.device)
    success_rate = 0
    # physical properties of the base averaged over one episode, e.g., velocity
    episode_base_lin_vel = torch.zeros(env.num_agents, 3, device=cfg.device)
    episode_base_ang_vel = torch.zeros(env.num_agents, 3, device=cfg.device)
    # physical properties of the base averaged over multiple environments (episodes)
    avg_base_lin_vel = torch.zeros(env.num_agents, 3, device=cfg.device)
    avg_base_ang_vel = torch.zeros(env.num_agents, 3, device=cfg.device)
    # individual returns averaged over multiple environments (episodes) 
    avg_individual_returns = torch.zeros(env.num_agents, len(env.reward_fns), device=cfg.device)

    start_time = time.time()

    state = env.reset()
    for i in range(total_num_samples):
        action, action_log_prob, state_value = actor_critic_net.evaluate(state, stochastic=True)
        action_mu = actor_critic_net.action_dist.mean
        action_sigma = actor_critic_net.action_dist.stddev
        next_state, reward, done, timeout = env.step(action.detach().clone())
        buffer.add_sample(state, action.detach(), action_mu.detach(), 
                        action_sigma.detach(), next_state, 
                        reward, done, action_log_prob.detach(), 
                        state_value.detach())
        state = next_state.clone()
        done_id = torch.where(torch.logical_or(timeout == True, done == True))[0]
        total_reward += reward

        # increment the physical properties
        episode_base_lin_vel += env.projected_lin_vel
        episode_base_ang_vel += env.projected_ang_vel
        
        # todo: save the physical statistics
        if buffer.full():
            ppo.compute_returns(next_state)
            pg_loss, value_loss, total_loss, lr = ppo.update()
            num_ppo_updates += 1
            terrain_context.epoch += 1
            buffer.clear()
            # log the statistics
            env_ids_with_ended_episodes = torch.where(num_episodes_per_env > 0)[0]
            # log the statistics
            if env_ids_with_ended_episodes.size(0) > 0 and num_ppo_updates % cfg.rl.log_eval_interval == 0:
                action_std = actor_critic_net.logstd.exp()
                avg_episodic_return = (total_episodic_returns[env_ids_with_ended_episodes] / num_episodes_per_env[env_ids_with_ended_episodes]).mean().item() 
                
                success_rates[env_ids_with_ended_episodes] = num_successful_runs[env_ids_with_ended_episodes] / num_episodes_per_env[env_ids_with_ended_episodes]
                success_rate = (success_rates[env_ids_with_ended_episodes]).mean().item()
                failure_rate = (num_unsuccessful_runs[env_ids_with_ended_episodes] / num_episodes_per_env[env_ids_with_ended_episodes]).mean().item()

                episodes_per_env_done = num_episodes_per_env[env_ids_with_ended_episodes].view(-1, 1)
                avg_returns = (avg_individual_returns[env_ids_with_ended_episodes] / episodes_per_env_done).mean(dim=0).cpu().numpy()
                # Create a dictionary to store all reward function returns for the epoch
                reward_logs = {}
                for i in range(len(env.reward_fns)):
                    reward_fn_name = env.reward_fns[i].__name__
                    individual_returns = avg_returns[i].item()
                    reward_logs[reward_fn_name] = individual_returns

                # reset the episodic statistics for the logged environments 
                num_episodes_per_env[env_ids_with_ended_episodes] = 0.0 
                num_unsuccessful_runs[env_ids_with_ended_episodes] = 0.0 
                num_successful_runs[env_ids_with_ended_episodes] = 0.0 
                total_episodic_returns[env_ids_with_ended_episodes] = 0.0
                avg_base_lin_vel[env_ids_with_ended_episodes] = 0.0
                avg_base_ang_vel[env_ids_with_ended_episodes] = 0.0
                avg_individual_returns[env_ids_with_ended_episodes] = 0.0

                # save all the model runs
                model_name = "model/{}/{}.pt".format(args.method, terrain_context.epoch)
                os.makedirs("model/{}".format(args.method), exist_ok=True)
                torch.save(actor_critic_net.state_dict(), model_name)

                # Log the table
                log_data = {
                    'Time elapsed (mins)': (time.time() - start_time) / 60.,
                    'Total Time elapsed (mins)': (time.time() - terrain_context.start_time) / 60.,
                    'Episode length': cfg.env.max_episode_length,
                    'Samples visited/Total samples': f'{num_samples_visited}/{total_num_samples}',
                    '# updates/Total updates': f'{(terrain_context.epoch)}/{num_learning_steps}',
                    'Avg. episodic return': avg_episodic_return,
                    'Avg. success rate': success_rate,
                    'Num. ended episodes': env_ids_with_ended_episodes.size(0),
                    'Avg. failure rate': failure_rate,
                    'PG loss': pg_loss,
                    'Value loss': value_loss,
                    'Total loss': total_loss,
                    'Learning rate': lr,
                    'Action std v': action_std[0].item(),
                    'Action std w': action_std[1].item(),
                    **reward_logs,
                    'Model Name': model_name,
                    'Environment': args.tif_name
                }

                # Log the data to WandB
                if args.wandb:
                    wandb.log(log_data)
                # Print a PrettyTable
                # os.system('clear')
                print('================= TRAIN =================')
                table = PrettyTable()
                table.field_names = ["Metric", "Value"]
                # Add log data to the table
                for key, value in log_data.items():
                    table.add_row([key, value])
                print(table)
        
        if (success_rate <= 0.99 and success_rate >= 0.6 and env.max_goal_dis > 5.99)\
            or num_ppo_updates >= 10\
            or terrain_context.epoch >= cfg.rl.num_learning_steps:
            # Compute the mean of non-zero elements
            mask = torch.zeros_like(success_rates, dtype=torch.bool)
            mask[env_ids_with_ended_episodes] = True
            S_masked = success_rates * mask.float()
            S_masked = S_masked.view(-1, cfg.env.num_agents_per_terrain)
            envs_success_rate = S_masked.sum(dim=1) / \
            (mask.view(-1, cfg.env.num_agents_per_terrain).sum(dim=1).clamp(min=1))
            envs_success_rate = envs_success_rate.view(-1).cpu().numpy()

            terrain_context.update(envs_success_rate, terrain_context.epoch)
            r.set('terrain_context', json.dumps(terrain_context.to_dict()))
            print("!!!break!!! ")
            break
        if success_rate <= 0.99 and success_rate >= 0.6:
            if env.max_goal_dis < 6.:
                env.max_goal_dis = min(6., env.max_goal_dis + 0.5)
                env.max_goal_ang = min(np.pi, env.max_goal_ang + np.pi / 18.)
                env.cfg.env.max_episode_length = env.max_goal_dis / 0.6 / env.cfg.sim.dt

        if done_id.size(0) > 0:
            num_episodes_per_env[done_id] += 1
            goal_terminations = env.goal_termination
            collision_terminations = env.collision_termination
            
            state = env.reset(done_id)
            # increment the success and failure counters
            num_unsuccessful_runs[torch.where(collision_terminations == 1.0)[0]] += 1
            num_successful_runs[torch.where(goal_terminations == 1.0)[0]] += 1
            # add the cumulative reward to episodic return 
            total_episodic_returns[done_id] += total_reward[done_id]
            
            steps = env.step_index[done_id].clone()
            steps = steps.view(-1, 1).repeat(1, 3)
            avg_base_lin_vel[done_id] += episode_base_lin_vel[done_id] / steps 
            avg_base_ang_vel[done_id] += episode_base_ang_vel[done_id] / steps 
            # reset the episode statistics
            episode_base_lin_vel[done_id] = torch.zeros_like(episode_base_lin_vel[done_id]) 
            episode_base_ang_vel[done_id] = torch.zeros_like(episode_base_ang_vel[done_id]) 

            avg_individual_returns[done_id] += env.individual_returns[done_id]
            env.individual_returns[done_id] = torch.zeros_like(env.individual_returns[done_id]) 

            # refresh the reward
            total_reward[done_id] = torch.zeros_like(done_id, device=cfg.device).float()
        
        envs_success_rate = None
        num_samples_visited = i * env.num_agents

if __name__ == "__main__":
    train()
