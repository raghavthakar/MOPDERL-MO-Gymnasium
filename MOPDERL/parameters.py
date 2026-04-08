import pprint
import torch
import os


class Parameters:
    def __init__(self, param, init=True):
        if not init:
            return
        param = param.parse_args()

        # Set the device to run on CUDA or CPU
        if not param.disable_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # Render episodes
        self.env_name = param.env

        self.eval_frames = 750
        self.num_objectives = 2
        if self.env_name == "MO-Hopper-v3":
            self.num_objectives = 3
        # Number of generations to Run
        if param.env == 'mo-hopper-2obj-v5' or param.env == 'MO-Hopper-v3':
            self.warm_up_frames = 4500000
            self.max_frames = 8000000
        elif param.env == 'mo-ant-2obj-v5' or param.env == 'mo-halfcheetah-v5':
            self.warm_up_frames = 4000000
            self.max_frames = 8000000
        elif param.env == 'mo-walker2d-v5':
            self.warm_up_frames = 5000000
            self.max_frames = 10000000
        else:
            self.warm_up_frames = 1500000
            self.max_frames = 4000000
        if param.warm_up is not None:
            self.warm_up_frames = param.warm_up
        if param.max_frames is not None:
            self.max_frames = param.max_frames


        # Synchronization
        if param.env == 'mo-hopper-2obj-v5' or param.env == 'mo-ant-2obj-v5' or param.env == 'mo-walker2d-v5':
            self.rl_to_ea_synch_period = 1
        else:
            self.rl_to_ea_synch_period = 10

        # Overwrite sync from command line if value is passed
        # if param.sync_period is not None:
        #     self.rl_to_ea_synch_period = param.sync_period
        # This repository runs in *boundary-only* mode.
        # Non-one-hot scalarizations are explicitly disabled.
        self.boundary_only = True
        self.num_rl_agents = self.num_objectives
        self.rl_type = param.rl_type


        # DDPG params
        self.use_ln = True
        self.gamma = 0.99
        self.tau = 0.001
        self.seed = param.seed
        self.batch_size = 128
        self.frac_frames_train = 1.0
        self.use_done_mask = True
        self.buffer_size = 1000000
        self.ls = 64

        #TD3 params
        self.actor_update_interval = 2

        # ========================================== NeuroEvolution Params =============================================

        # Num of trials
        if param.env == 'mo-hopper-2obj-v5' or param.env == 'MO-Reacher-v2':
            self.num_evals = 3
        elif param.env == 'mo-walker2d-v5':
            self.num_evals = 5
        else:
            self.num_evals = 1

        # Elitism Rate
        if param.env == 'MO-Reacher-v2' or param.env == 'mo-walker2d-v5' or param.env == 'mo-ant-2obj-v5' or param.env == 'mo-hopper-2obj-v5':
            self.elite_fraction = 0.2
        else:
            self.elite_fraction = 0.1

        # Number of actors in the population
        self.pop_size = param.num_individuals * self.num_rl_agents
        self.each_pop_size = param.num_individuals
        self.max_child = self.pop_size
        self.count_actors = 0


        # Priority magnitude
        self.priority = param.priority_mag

        # Mutation and crossover
        self.crossover_prob = 0.0
        self.mutation_prob = 0.9
        self.mutation_mag = param.mut_mag
        self.mutation_noise = param.mut_noise
        self.mutation_batch_size = 256
        # Genetic memory size
        self.individual_bs = 8000

        # Save Results
        self.state_dim = None  # To be initialised externally
        self.action_dim = None  # To be initialised externally
        self.save_foldername = param.logdir
        self.run_id = param.run_id
        if not os.path.exists(self.save_foldername):
            os.makedirs(self.save_foldername)
        
        self.wandb = not param.disable_wandb
        self.checkpoint = param.checkpoint
        self.checkpoint_id = param.checkpoint_id
        self.save_ckpt_period = param.save_ckpt

        # ========================================== Weightt conditioning Params =============================================
        self.weight_conditioned = param.weight_conditioned
        self.secondary_critics = param.secondary_critics
        
    def write_params(self, path, stdout=True):
        # Dump all the hyper-parameters in a file.
        params = pprint.pformat(vars(self), indent=4)
        if stdout:
            print(params)

        with open(os.path.join(path, 'info.txt'), 'a') as f:
            f.write(params)