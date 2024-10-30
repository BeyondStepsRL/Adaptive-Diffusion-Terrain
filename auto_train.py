import subprocess
import redis
from contexts.terrain_context import TerrainContextSpace
import json
import wandb
import time
import os
import torch
from cfg.jackal_config import JackalCfg

def retrieve_context(r):
    terrain_context_data = r.get('terrain_context')
    if terrain_context_data is None:
        raise ValueError("No terrain context found in Redis!")
    terrain_context_dict = json.loads(terrain_context_data)
    terrain_context = TerrainContextSpace.from_dict(terrain_context_dict)
    return terrain_context

def load_from_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found.")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return TerrainContextSpace.from_dict(data)

def choose(choices):
    while True:
        print("Please choose a project:")
        for i, name in enumerate(choices):
            print(f"{i + 1}. {name}")
        try:
            choice = int(input("Enter the number corresponding to the project: ")) - 1
            # Check if the choice is valid
            if 0 <= choice < len(choices):
                choice = choices[choice]
                print(f"Running with project name: {choice}")
                return choice
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number corresponding to the project.")

# Define the available project names
trainers = [
    'teacher',
    'student'
]
environment_samplers = [
    'nature',
    'procedural',
    'ddpm',
]
environment_curricula = [
    'uniform',
    'adaptive'
]

trainer = choose(trainers)
context_space = JackalCfg()
context_space.context.environment.sampler = choose(environment_samplers)
context_space.context.environment.curriculum = choose(environment_curricula)

# Prompt the user for a file input
file_path = input("Please enter the file path (press Enter to skip): ")
if not file_path:
    file_path = None

project = 'Offroad Navigation ' + trainer
method = context_space.context.environment.sampler + ' ' + context_space.context.environment.curriculum

if trainer == 'teacher':
    if file_path is not None:
        terrain_context = load_from_file(file_path)
    else:
        terrain_context = TerrainContextSpace(context_space)
    terrain_context.start_time = time.time()
    terrain_context.flag_evaluaton = False
    policy = 'rl_agents.train_teacher'
elif trainer == 'student':
    if file_path is None:
        file_path = f"wandb/Log/{context_space.context.environment.sampler}/{context_space.context.environment.curriculum}/latest.json"
    terrain_context = load_from_file(file_path)
    terrain_context.num_rows = context_space.context.environment.num_rows_distill
    terrain_context.num_cols = context_space.context.environment.num_cols_distill
    terrain_context.num_terrains = terrain_context.num_rows * terrain_context.num_cols
    num_distill_steps = len(terrain_context.data) // terrain_context.num_terrains
    policy = 'bc_agents.train_student'

run_id = wandb.util.generate_id()
# Connect to the Redis server
r = redis.Redis(host='localhost', port=6379, db=0)

# Create the list of programs to run
for itr in range(context_space.rl.num_learning_steps):
    slice_start = None if trainer == 'teacher' else itr
    tif_file = terrain_context.sample(slice_start)
    backbone = terrain_context.get_backbone()
    r.set('terrain_context', json.dumps(terrain_context.to_dict()))
    torch.cuda.empty_cache()
    
    program = [
        'python3', '-m', policy,
        '--tif_name', tif_file,
        '--backbone', backbone,
        '--project', project,
        '--method', method,
        '--wandb', 'False',
        '--wandb_entity', 'Nullptr',
        '--wandb_id', run_id
    ]

    print(f"Executing: {program}")
    subprocess.call(program)
    
    terrain_context_data = r.get('terrain_context')
    if terrain_context_data is None:
        raise ValueError("No terrain context found in Redis!")
    terrain_context_dict = json.loads(terrain_context_data)
    terrain_context = TerrainContextSpace.from_dict(terrain_context_dict)

    if trainer == 'teacher':
        terrain_context.evaluate(r)

    terrain_context.save_to_file()

    if trainer == 'teacher' and terrain_context.epoch >= context_space.rl.num_learning_steps:
        break
    elif trainer == 'student' and itr >= num_distill_steps:
        break