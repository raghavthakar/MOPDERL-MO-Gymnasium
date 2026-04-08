import os
from pathlib import Path

# Seed env vars as early as possible (before importing torch) for determinism.
from .seed import seed_everything

from datetime import datetime
import numpy as np
import time
import argparse
import logging

import mo_gymnasium as gym
import mo_gymnasium.envs.mujoco  # Importing this module registers the environments
import torch
import wandb

from .seed import seed_torch, seed_env
from .parameters import Parameters
from . import mo_agent
from . import utils

parser = argparse.ArgumentParser()
# Updated environment choices to reflect mo-gymnasium names
parser.add_argument('-env', help='Environment Choices: (mo-swimmer-v5) (mo-halfcheetah-v5) (mo-hopper-2obj-v5) ' +
                                 '(mo-walker2d-v5) (mo-ant-2obj-v5)', required=True, type=str)
parser.add_argument('-seed', help='Random seed to be used', type=int, required=True)
parser.add_argument('-disable_cuda', help='Disables CUDA', action='store_true')
parser.add_argument('-mut_mag', help='The magnitude of the mutation', type=float, default=0.05)
parser.add_argument('-mut_noise', help='Use a random mutation magnitude', action='store_true')
parser.add_argument('-logdir', help='Folder where to save results', type=str, required=True)
parser.add_argument('-warm_up', help='Warm up frames', type=int)
parser.add_argument('-max_frames', help='Max frames', type=int)
parser.add_argument('-num_individuals', help='Number of individual per pderl population', type=int, default=10)
parser.add_argument('-num_generations', help='Max number of generation', type=int)
parser.add_argument('-priority_mag', help='Percent of priority for objective', type=float, default=1.0)
parser.add_argument('-rl_type', help='Type of rl-agents', type=str, default="ddpg")
parser.add_argument('-checkpoint', help='Load checkpoint', action='store_true')
parser.add_argument('-checkpoint_id', help='Select -run- to load checkpoint', type=int)
parser.add_argument('-run_id', help="Specify run id, if not given, get id as len(run)", type=int)
parser.add_argument('-save_ckpt', help="Save checkpoint every _ step, 0 for no save", type=int, default=1)
parser.add_argument('-disable_wandb', action="store_true", default=False)
# boundary_only is enforced by the codebase (one-hot weights only). Kept for
# backward compatibility with older scripts.
parser.add_argument('-boundary_only', action='store_true', default=True)
parser.add_argument('-weight_conditioned', action='store_true', default=False)
parser.add_argument('-secondary_critics', action='store_true', default=False)

# Updated map with correct mo-gymnasium environment names and versions
name_map = {
    'MO-Swimmer-v2': 'mo-swimmer-v5',
    'MO-HalfCheetah-v2': 'mo-halfcheetah-v5',
    'MO-Hopper-v2': 'mo-hopper-2obj-v5',
    'MO-Walker2d-v2': 'mo-walker2d-v5',
    'MO-Ant-v2': 'mo-ant-2obj-v5',
}

if __name__ == "__main__":
    # Parse args and construct the parameters object first.
    parameters = Parameters(parser)

    # Deterministic seeding (must happen before any model/env stochasticity).
    seed_everything(parameters.seed)
    seed_torch(parameters.seed)

    if not os.path.exists(parameters.save_foldername):
        os.mkdir(parameters.save_foldername)
    env_folder = os.path.join(parameters.save_foldername, parameters.env_name)
    if not os.path.exists(env_folder):
        os.mkdir(env_folder)
    list_run = sorted(os.listdir(env_folder))
    if parameters.checkpoint:
        if parameters.checkpoint_id is not None:
            run_folder = os.path.join(env_folder, "run_"+str(parameters.checkpoint_id))
        else:
            run_folder = os.path.join(env_folder, list_run[-1])
    else:
        run_id = "run_"+str(len(list_run))
        if parameters.run_id is not None:
            run_id = "run_"+str(parameters.run_id)
        run_folder = os.path.join(env_folder, run_id)
    if not os.path.exists(run_folder):    
        os.mkdir(run_folder)    

    if parameters.wandb: wandb.init(project=parameters.env_name, entity="mopderl", id=str(Path(run_folder).name), resume=parameters.checkpoint) 
    logging.basicConfig(filename=os.path.join(run_folder, "logger.log"),
                        format=('[%(asctime)s] - '
                                '[%(levelname)4s]:\t'
                                '%(message)s'
                                '\t(%(filename)s:'
                                '%(funcName)s():'
                                '%(lineno)d)\t'),
                        filemode='a',
                        level=logging.DEBUG)
    logger = logging.getLogger()
    logger.info("Start time: " + datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

    # Create Env
    env = utils.NormalizedActions(gym.make(parameters.env_name))
    seed_env(env, parameters.seed)
    parameters.action_dim = env.action_space.shape[0]
    parameters.state_dim = env.observation_space.shape[0]

    # Write the parameters to a the info file and print them
    parameters.write_params(path=run_folder)

    # Create Agent
    reward_keys_path = Path(__file__).resolve().parent / "reward_keys.json"
    reward_keys = utils.parse_json(str(reward_keys_path))[parameters.env_name]
    agent = mo_agent.MOAgent(parameters, env, reward_keys, run_folder)
    print('Running', parameters.env_name, ' State_dim:', parameters.state_dim, ' Action_dim:', parameters.action_dim)
    logger.info('Running' + str(parameters.env_name) + ' State_dim:' + str(parameters.state_dim) + ' Action_dim:' + str(parameters.action_dim))
    logger.info("Priority: " + str(parameters.priority))

    time_start = time.time()
    warm_up_saved = False
    last_saved_total_frames = -1

    while np.sum(agent.num_frames < agent.max_frames).astype(int) > 0:
        logger.info("************************************************")
        logger.info("\t\tGeneration: " + str(agent.iterations))
        logger.info("************************************************")
        stats_wandb = agent.train_final(logger)

        if parameters.wandb and len(stats_wandb):
            current_pareto = stats_wandb.pop("pareto")
            current_pareto = [list(point) for point in current_pareto]
            table = wandb.Table(data=current_pareto, columns=reward_keys)
            wandb.log({ 
                **{"Current pareto front" : wandb.plot.scatter(table, reward_keys[0], reward_keys[1], title="Current pareto front")}, 
                **stats_wandb
            })

        print('#Generation:', agent.iterations, '#Frames:', agent.num_frames,
              ' ENV:  '+ parameters.env_name)
        print()
        logger.info("\n\n")
        logger.info("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        logger.info("=>>>>>> Num frames: " + str(agent.num_frames))
        logger.info("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

        # ---------------- Checkpointing ----------------
        # -save_ckpt is interpreted as a *frame period* (based on sum(num_frames)).
        # This ensures consistent checkpoints irrespective of generation length.
        total_frames = int(np.sum(agent.num_frames))
        should_save_periodic = (
            parameters.save_ckpt_period > 0
            and (last_saved_total_frames < 0 or (total_frames - last_saved_total_frames) >= parameters.save_ckpt_period)
        )
        should_save_end = np.sum(agent.num_frames < agent.max_frames).astype(int) == 0

        if should_save_periodic or should_save_end:
            # Save into phase-specific latest folders (warm_up_latest or stage2_latest)
            # and keep checkpoint/latest pointing to whichever phase is currently active.
            ckpt_folder = agent.get_active_checkpoint_folder() if hasattr(agent, "get_active_checkpoint_folder") else None
            agent.save_info(checkpoint_folder=ckpt_folder)
            if hasattr(agent, "update_latest_checkpoint_pointer"):
                agent.update_latest_checkpoint_pointer()
            last_saved_total_frames = total_frames
            logger.info("Saved checkpoint successfully!\n\n")

        # Preserve warm-up-final checkpoint at transition (mo_agent handles this)
        if not warm_up_saved and np.sum(agent.num_frames < parameters.warm_up_frames).astype(int) == 0:
            if hasattr(agent, "warmup_final_saved") and agent.warmup_final_saved:
                logger.info("Warm-up final checkpoint frozen successfully.\n\n")
            warm_up_saved = True

        if len(stats_wandb):
            agent.archive.save_info()
        
    logger.info("End time: " + datetime.now().strftime("%d/%m/%Y %H:%M:%S"))