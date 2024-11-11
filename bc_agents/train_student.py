from contexts.offroad_env import OffRoadEnv
from contexts.terrain_context import TerrainContextSpace
import numpy as np
import torch
from torch.optim import AdamW
from rl_agents.actor_critic_net import ActorCriticNet
from cfg.jackal_config import JackalCfg
from bc_agents.student_net import LearnerNet, transform_depth_image, EarlyStopScheduler
import time
import argparse
import wandb
import redis
import json
import os
import sys
from prettytable import PrettyTable
from bc_agents.dataset import BCDataset
from PIL import Image
import pandas as pd
from tqdm import tqdm


def train_n_epoch(model, criterion, optimizer, trainloader, valloader):
    pbar = tqdm(total=len(trainloader))
    scheduler = EarlyStopScheduler(optimizer, factor=0.1, verbose=True, min_lr=0.000001, patience=4)
    model.train()
    device = next(model.parameters()).device
    best_loss = 999999

    for epoch in range(200):
        train_loss = 0
        val_loss = 0

        for i, data in enumerate(trainloader):
            states = data['student_state'].to(device)
            depths = data['observation'].to(device) / 255.
            depths = transform_depth_image(depths)
            labels = data['expert_action'].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs, hx = model(depths, states)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss /= (i + 1)

        for i, data in enumerate(valloader):
            states = data['student_state'].to(device)
            depths = data['observation'].to(device) / 255.
            depths = transform_depth_image(depths)
            labels = data['expert_action'].to(device)

            # forward + backward + optimize
            outputs, hx = model(depths, states)
            loss = criterion(outputs, labels)

            # print statistics
            val_loss += loss.item()
        val_loss /= (i + 1)

        if val_loss < best_loss:
            best_loss = val_loss
        
        pbar.set_description(f"loss train={train_loss:0.4g}  val={val_loss:0.4g}")
        if scheduler.step(val_loss):
            pbar.set_description('Early Stopping!')
            pbar.update(1)
            break
        pbar.update(1)
    
    pbar.close()

def run_closed_loop(env, teacher, learner, data_path, num_episodes):
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(data_path+'depth/', exist_ok=True)
    df = pd.DataFrame(columns=('agent_id', 'image_name', 'traj_id', 'success', 
                               'student_state', 'student_action', 'expert_action'))

    state, student_state, observation = env.reset()

    hx = None  # Hidden state of the RNN

    with torch.no_grad():
        traj_ids = torch.zeros(env.num_agents, device=env.device)
        while True:
            # After sufficient imitation bootstrapping, move to DAgger
            # encoded_obs = transform_depth_image(observation)
            # pred, hx = learner(encoded_obs, student_state, hx)
            # next_state, student_state, observation, reward, done, timeout = env.step(pred)

            action, _, _ = teacher.evaluate(next_state, stochastic=False)
            pred = action
            next_state, student_state, observation, reward, done, timeout = env.step(action)
            label = env.applied_actions ## action

            t = time.strftime("%Y%m%d-%H%M%S")
            for agent_id in range(env.num_agents):
                image_name = data_path + 'depth/' + "{}-{}.png".format(agent_id, t)
                _image = (observation[agent_id] * 255.).cpu().numpy()
                _image = Image.fromarray(_image.astype(np.uint8), mode='L')
                _image.save(image_name)
                
                df = pd.concat([df, pd.DataFrame.from_dict({
                                'agent_id': agent_id,
                                'traj_id': traj_ids[agent_id].item(),
                                'success': 0,
                                'image_name': image_name,
                                'student_state': student_state[agent_id].cpu().numpy(),
                                'student_action': pred[agent_id].cpu().numpy(), 
                                'expert_action': label[agent_id].cpu().numpy()}, orient='index').T])
            
            done_id = torch.where(torch.logical_or(timeout == True, done == True))[0]
            if done_id.size(0) > 0:
                goal_terminations = env.goal_termination
                print(torch.sum(goal_terminations))
                
                success_agents = torch.where(goal_terminations == 1.0)[0]
                for agent_id in success_agents:
                    traj_id = traj_ids[agent_id].item()
                    df.loc[(df['agent_id'] == agent_id.item()) & (df['traj_id'] == traj_id), 'success'] = 1

                traj_ids[success_agents] += 1

                next_state, student_state, observation = env.reset(done_id)
                hx = None
                
                if success_agents.size(0) > 0:
                    num_episodes = num_episodes - 1
                    if num_episodes == 0:
                        df.to_csv(data_path + "data.csv", index=False, header=True)
                        return

def train_student():
    parser = argparse.ArgumentParser(description="Train Student Model")

    parser.add_argument('--tif_name', type=str, default="Nullptr", help='tif name for terrains')
    parser.add_argument('--backbone', type=str, default="Nullptr", help='model to load')
    parser.add_argument('--project', type=str, default="Offroad Navigation")
    parser.add_argument('--method', type=str, default="NatureTerrain")

    parser.add_argument('--wandb', type=bool, default=False, help='https://wandb.ai')
    parser.add_argument('--wandb_entity', type=str, default="Nullptr")
    parser.add_argument('--wandb_id', type=str, default="Nullptr")

    args = parser.parse_args()

    wandb.init(project=args.project, entity=args.wandb_entity, name=args.method, id=args.wandb_id, resume="allow")

    cfg = JackalCfg()
    cfg.robot.camera.use_camera = True
    cfg.state_space.use_noise = True
    cfg.state_space.use_globalmap = True
    cfg.env.num_agents_per_terrain = cfg.env.num_agents_per_terrain_distill
    cfg.context.environment.num_rows = cfg.context.environment.num_rows_distill
    cfg.context.environment.num_cols = cfg.context.environment.num_cols_distill
    
    # Redis
    r = redis.Redis(host='localhost', port=6379, db=0)
    terrain_context_data = r.get('terrain_context')
    if terrain_context_data is None:
        raise ValueError("No terrain context found in Redis!")
    terrain_context_dict = json.loads(terrain_context_data)
    terrain_context = TerrainContextSpace.from_dict(terrain_context_dict)

    env = OffRoadEnv(cfg, mode='rl', render=False, terrain_path=args.tif_name)
    env.max_goal_dis = 6.0
    env.max_goal_ang = np.pi
    env.cfg.env.max_episode_length = env.max_goal_dis / 0.6 / env.cfg.sim.dt

    state_dim = env.state_dim
    action_dim = env.action_dim
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
        actor_critic_net.eval()
    else:
        exit(1)

    # behavior cloning agent
    bc_agent = LearnerNet(action_lower=cfg.action_space.action_lower, 
                          action_upper=cfg.action_space.action_upper).to(device=cfg.device)
    if os.path.exists("model/ncp_agent.pt"):
        bc_agent.load_state_dict(torch.load("model/ncp_agent.pt"))
    
    optimizer = AdamW(bc_agent.parameters(), lr=0.0001, weight_decay=0.001)
    mse_loss = torch.nn.MSELoss()
    bc_agent.train()

    for epoch in range(1):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        data_path = f"wandb/Log/Student/{args.method}_{timestamp}/"

        run_closed_loop(env, actor_critic_net, bc_agent, data_path, num_episodes=10)

        train_ds = BCDataset(data_path+'data.csv', split='train')
        trainloader = torch.utils.data.DataLoader(
            train_ds, batch_size=32, num_workers=4, shuffle=True
        )

        val_ds = BCDataset(data_path+'data.csv', split='val')
        valloader = torch.utils.data.DataLoader(
            val_ds, batch_size=32, num_workers=4, shuffle=True
        )

        train_n_epoch(bc_agent, mse_loss, optimizer, trainloader, valloader)
        
        torch.save(bc_agent.state_dict(), "model/ncp_agent.pt")

if __name__ == "__main__":
    train_student()
