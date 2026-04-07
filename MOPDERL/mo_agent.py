from .parameters import Parameters
import numpy as np
import random
import torch
from . import ddpg
from .td3 import TD3
import os, shutil
from typing import Optional
from .pderl_tools import PDERLTool
from .nsga2_tools import NSGA, nsga2_sort
from .archive import *
from .utils import create_scalar_list

class MOAgent:
    def __init__(self, args: Parameters, env, reward_keys: list, run_folder) -> None:
        self.args = args
        self.env = env
        self.reward_keys = reward_keys
        self.init_env_folder(run_folder)

        self.num_objectives = args.num_objectives
        self.num_rl_agents = args.num_rl_agents

        # self.pop_individual_type = [int(i / int(args.pop_size / args.num_rl_agents)) for i in range(args.pop_size)]

        self.rl_agents = []
        # for i in range(args.num_rl_agents):
        #     scalar_weight = np.ones(args.num_rl_agents) * (1-self.args.priority)/(args.num_rl_agents-1)
        #     scalar_weight[i] = self.args.priority
        #     if args.rl_type == "ddpg":
        #         self.rl_agents.append(ddpg.DDPG(args, scalar_weight=scalar_weight))
        #     elif args.rl_type == "td3":
        #         self.rl_agents.append(TD3(args, scalar_weight=scalar_weight))
        #     else:
        #         raise NotImplementedError("Unknown rl agent type, must be ddpg or td3, got " + args.rl_type)
        
        self.each_pop_size = int(args.pop_size/args.num_rl_agents)
        scalar_weight_list = create_scalar_list(self.num_objectives, self.args.boundary_only) #NOTE: obj_num can be recovered via argmax on the one-hot vector
        self.pop_individual_type = []
        for i in range(len(scalar_weight_list)):
            for _ in range(self.each_pop_size):
                self.pop_individual_type.append(i)
        for w_idx, weight in enumerate(scalar_weight_list):
            if args.rl_type == "ddpg":
                # Get all elements except w_idx
                other_weights = np.concatenate((scalar_weight_list[:w_idx], scalar_weight_list[w_idx+1:]), axis=0)
                self.rl_agents.append(ddpg.DDPG(args, scalar_weight=weight, other_weights=other_weights))
            elif args.rl_type == "td3":
                self.rl_agents.append(TD3(args, scalar_weight=weight))
            else:
                raise NotImplementedError("Unknown rl agent type, must be ddpg or td3, got " + args.rl_type)
        
        # reseed to ensure identical population inits across runs
        torch.manual_seed(args.seed + 100) 
        np.random.seed(args.seed + 100)
        random.seed(args.seed + 100)

        self.max_frames = args.max_frames
        self.num_frames = np.zeros(args.num_rl_agents)
        self.iterations = 0
        self.num_games = 0
        self.gen_frames = np.zeros_like(self.num_frames)
        self.trained_frames = np.zeros_like(self.num_frames)
        self.fitness = np.zeros((args.pop_size, self.num_objectives))
        self.pop = [] # store actors in the second stage
        
        self.fitness_list = [np.zeros((self.each_pop_size, self.num_objectives)) for _ in range(self.num_rl_agents)]
        self.pop_list = [] # store actors in the first stage
        
        self.warm_up = True
        for _ in range(args.num_rl_agents):
            temp_pop = []
            for _ in range(self.each_pop_size):
                temp_pop.append(ddpg.GeneticAgent(args))
            self.pop_list.append(temp_pop)
        self.pderl_tools = PDERLTool(args, self.rl_agents, self.evaluate)
        self.nsga = NSGA(args, self.rl_agents, self.evaluate)
        self.archive = Archive(args, self.archive_folder)

        if args.checkpoint:
            print("Loading info...")
            self.load_info()
            print("*" * 10)
            print("Load info sucessfully!!!")
            print("*" * 10)
    
    def init_env_folder(self, run_folder):
        self.run_folder = run_folder
        if not os.path.exists(self.run_folder):
            os.mkdir(self.run_folder)

        # --- Checkpoint layout ---
        # We keep the internal structure identical to the original codebase
        # (info.npy + warm_up/ or multiobjective/ subfolders), but store warm-up
        # and stage-2 checkpoints in separate top-level directories to avoid
        # overwriting the warm-up checkpoint at the transition.
        self.checkpoint_root = os.path.join(self.run_folder, "checkpoint")
        os.makedirs(self.checkpoint_root, exist_ok=True)

        # New layout (preferred)
        self.checkpoint_warmup_latest = os.path.join(self.checkpoint_root, "warm_up_latest")
        self.checkpoint_warmup_final = os.path.join(self.checkpoint_root, "warm_up_final")
        self.checkpoint_stage2_latest = os.path.join(self.checkpoint_root, "stage2_latest")
        self.checkpoint_latest_link = os.path.join(self.checkpoint_root, "latest")

        for p in [self.checkpoint_warmup_latest, self.checkpoint_warmup_final, self.checkpoint_stage2_latest]:
            os.makedirs(p, exist_ok=True)

        # Legacy layout compatibility: older runs may have info.npy directly under
        # checkpoint/. If so, we keep using that folder for load/save unless the
        # new layout already exists.
        legacy_info = os.path.join(self.checkpoint_root, "info.npy")
        new_layout_has_any = (
            os.path.exists(os.path.join(self.checkpoint_warmup_latest, "info.npy"))
            or os.path.exists(os.path.join(self.checkpoint_stage2_latest, "info.npy"))
            or os.path.exists(os.path.join(self.checkpoint_warmup_final, "info.npy"))
        )
        self.using_legacy_checkpoints = bool(os.path.exists(legacy_info) and not new_layout_has_any)

        if self.using_legacy_checkpoints:
            # Preserve old behavior
            self.checkpoint_folder = self.checkpoint_root
        else:
            # Choose the best existing target for the "latest" pointer.
            if os.path.exists(os.path.join(self.checkpoint_stage2_latest, "info.npy")):
                target = self.checkpoint_stage2_latest
            elif os.path.exists(os.path.join(self.checkpoint_warmup_latest, "info.npy")):
                target = self.checkpoint_warmup_latest
            else:
                target = self.checkpoint_warmup_latest

            self._set_latest_checkpoint_pointer(target)
            # Default read/write path for resume is always the latest pointer.
            self.checkpoint_folder = self.checkpoint_latest_link if os.path.exists(self.checkpoint_latest_link) else target

        # Whether we have already frozen the warm-up-final checkpoint
        self.warmup_final_saved = os.path.exists(os.path.join(self.checkpoint_warmup_final, "info.npy"))

        self.archive_folder = os.path.join(self.run_folder, "archive")
        if not os.path.exists(self.archive_folder):
            os.mkdir(self.archive_folder)

    def _set_latest_checkpoint_pointer(self, target_folder: str) -> None:
        """Create/overwrite the checkpoint/latest symlink to point at target_folder."""
        # If we're in legacy mode, do nothing.
        if getattr(self, "using_legacy_checkpoints", False):
            return

        try:
            # Remove existing link/file if present
            if os.path.islink(self.checkpoint_latest_link) or os.path.isfile(self.checkpoint_latest_link):
                os.remove(self.checkpoint_latest_link)
            # If it's a directory (not a symlink), we avoid deleting user data.
            if os.path.isdir(self.checkpoint_latest_link) and not os.path.islink(self.checkpoint_latest_link):
                return
            os.symlink(os.path.abspath(target_folder), self.checkpoint_latest_link)
        except OSError:
            # Symlinks can fail on some systems; fall back to no-link behavior.
            pass

    def get_active_checkpoint_folder(self) -> str:
        """Return the folder that should receive the latest checkpoint for the current phase."""
        if getattr(self, "using_legacy_checkpoints", False):
            return self.checkpoint_root
        return self.checkpoint_warmup_latest if self.warm_up else self.checkpoint_stage2_latest

    def update_latest_checkpoint_pointer(self) -> None:
        """Update checkpoint/latest to point to the active phase checkpoint."""
        if getattr(self, "using_legacy_checkpoints", False):
            return
        self._set_latest_checkpoint_pointer(self.get_active_checkpoint_folder())
        

    def evaluate(self, agent, is_render=False, is_action_noise=False,
             store_transition=True, rl_agent_index=None):
        eval_frames = self.args.eval_frames
        total_reward = np.zeros(len(self.reward_keys), dtype=np.float32)
        
        # CORRECTED: Unpack the state and info dictionary from reset()
        state, _ = self.env.reset()
        
        done = False
        cnt_frame = 0
        while not done:
            # if self.args.render and is_render: self.env.render()
            action = agent.actor.select_action(np.array(state), is_action_noise)

            # CORRECTED: Unpack all 5 return values from the modern Gymnasium API
            next_state, reward, terminated, truncated, info = self.env.step(action.flatten())
            
            # CORRECTED: The episode is done if it's terminated OR truncated
            done = terminated or truncated

            # CORRECTED: The 'reward' variable is the multi-objective reward vector.
            # The line `reward = info["obj"]` is no longer needed.
            total_reward += reward

            transition = (state, action, reward, next_state, float(done))
            if store_transition:
                if isinstance(agent, ddpg.GeneticAgent):
                    agent.yet_eval = True
                    if rl_agent_index is not None:
                        agent.buffer.add(*transition)
                        self.gen_frames[rl_agent_index] += 1
                        self.rl_agents[rl_agent_index].buffer.add(*transition)
                    else:
                        self.gen_frames += 1
                        agent.buffer.add(*transition)
                        for rl_agent in self.rl_agents:
                            rl_agent.buffer.add(*transition)
                elif isinstance(agent, ddpg.DDPG) or isinstance(agent, TD3):
                    self.gen_frames[rl_agent_index] += 1
                    agent.buffer.add(*transition)
                else:
                    raise NotImplementedError("Unknown agent class")

            state = next_state
            cnt_frame += 1
            if cnt_frame == eval_frames:
                break
        if store_transition: self.num_games += 1

        return total_reward
    
    def rl_to_evo(self, rl_agent: ddpg.DDPG or TD3, evo_net: ddpg.GeneticAgent):
        for target_param, param in zip(evo_net.actor.parameters(), rl_agent.actor.parameters()):
            target_param.data.copy_(param.data)
        evo_net.buffer.reset()
        evo_net.buffer.add_content_of(rl_agent.buffer)

    def train_rl_agents(self, logger):
        # print("================Begin training rl agents========================")
        logger.info("Begin training rl agents")
        actors_loss, critics_loss = [], []
        logger.info("Gen frames: " + str(self.gen_frames))
        for i, rl_agent in enumerate(self.rl_agents):
            if len(rl_agent.buffer) > self.args.batch_size * 5:
                print("Evaluating agent: ", i, int(self.gen_frames[i]*self.args.frac_frames_train))
                actor_loss = []
                critic_loss = []
                for _ in range(int(self.gen_frames[i] * self.args.frac_frames_train)):
                # for _ in range(15):
                    batch = rl_agent.buffer.sample(self.args.batch_size)
                    pgl, delta = rl_agent.update_parameters(batch)
                    actor_loss.append(pgl)
                    critic_loss.append(delta)
                actors_loss.append(np.mean(actor_loss))
                critics_loss.append(np.mean(critic_loss))
        self.num_frames += np.array(self.gen_frames)
        self.trained_frames += np.array(self.gen_frames * self.args.frac_frames_train, dtype=np.int32)
        self.gen_frames *= 0.0
        return 

    def flatten_list(self):
        for scalar_pop in self.pop_list:
            for actor in scalar_pop:
                self.pop.append(actor)

    def train_final(self, logger):
        self.iterations += 1
        logger.info("Begin mo-pderl training")

        stats_wandb = {}

        if self.warm_up:
            if np.sum((self.num_frames <= self.args.warm_up_frames).astype(np.int32)) == 0:
                # We are *about to* transition out of warm-up.
                # Freeze a final warm-up checkpoint (never overwritten) so it can be
                # used later to restart stage-2 training or to bootstrap other algorithms.
                if (not getattr(self, "using_legacy_checkpoints", False)) and (not self.warmup_final_saved):
                    self.save_info(checkpoint_folder=self.checkpoint_warmup_final, warm_up_override=True)
                    self.save_warm_up_info_file(logger, checkpoint_folder=self.checkpoint_warmup_final)
                    self.warmup_final_saved = True

                self.warm_up = False
                self.flatten_list()
                for i, genetic_agent in enumerate(self.pop):
                    for _ in range(self.args.num_evals):
                        episode_reward = self.evaluate(genetic_agent, is_render=False, is_action_noise=False, store_transition=True)
                        self.fitness[i] += episode_reward
                self.fitness /= self.args.num_evals
                logger.info("=>>>>>> Finish warming-up and flattening")
        # ========================== EVOLUTION  ==========================
        if self.warm_up:
            for rl_agent_id in range(self.num_rl_agents):
                pop = self.pop_list[rl_agent_id]
                fitness = np.zeros((self.each_pop_size, self.num_objectives))
                self.fitness_list[rl_agent_id] = fitness
                if self.num_frames[rl_agent_id] < self.args.warm_up_frames:
                    for i in range(self.each_pop_size):
                        for _ in range(self.args.num_evals):
                            # episode_reward = self.evaluate(pop[i], is_render=False, is_action_noise=False, store_transition=True, rl_agent_index=None)
                            episode_reward = self.evaluate(pop[i], is_render=False, is_action_noise=False, store_transition=True, rl_agent_index=rl_agent_id)
                            fitness[i] += episode_reward
                        fitness[i] /= self.args.num_evals
                    self.pderl_tools.pderl_step(pop, rl_agent_id, fitness, logger)
                
        else:
            sorted_pareto_fronts = nsga2_sort(fitness=self.fitness, max_point=1e6)
            self.fitness, stats = self.nsga.mopderl_step(self.archive, self.pop, self.fitness, self.pop_individual_type, sorted_pareto_fronts, self.num_frames, logger)
            stats_wandb = {**stats_wandb, **stats}
            stats_wandb["pareto"] = self.archive.fitness_np
            # new_sorted_pareto_fronts = nsga2_sort(fitness=self.fitness, max_point=1e6)
        # ========================== DDPG ===========================
        # Collect experience for training and testing rl-agents
        for i, agent in enumerate(self.rl_agents):
            if self.num_frames[i] < self.args.max_frames:
                self.evaluate(agent, is_action_noise=True, rl_agent_index=i)

        self.train_rl_agents(logger)

        # print("================Begin testing rl agents========================")
        logger.info("Testing rl agents (no store transition)")
        rl_agent_score = np.zeros((self.num_rl_agents, self.num_objectives))
        for i, agent in enumerate(self.rl_agents):
            for _ in range(3):
                episode_reward = self.evaluate(agent, store_transition=False, is_action_noise=False)
                rl_agent_score[i] += episode_reward
        rl_agent_score /= 3
        
        if self.iterations % self.args.rl_to_ea_synch_period == 0 and self.warm_up:
            for rl_agent_id in range(self.num_rl_agents):
                scalar_fitness = np.dot(self.fitness_list[rl_agent_id], self.rl_agents[rl_agent_id].scalar_weight) 
                index_to_replace = np.argmin(scalar_fitness)
                self.rl_to_evo(self.rl_agents[rl_agent_id], self.pop_list[rl_agent_id][index_to_replace])
            logger.info("Sync from RL ---> Nevo")

        # Calculate num points in every front
        # num_in_front = [len(front) for front in new_sorted_pareto_fronts] if not self.warm_up else 0

        # pareto_first_front = new_sorted_pareto_fronts[0] if not self.warm_up else 0

        # pareto_first_front_type = np.array(self.pop_individual_type)[pareto_first_front] if not self.warm_up else 0

        # -------------------------- Collect statistics --------------------------
        return stats_wandb

    def save_info_mo(self, folder_path):
        rl_agents_folder = os.path.join(folder_path, "rl_agents")
        if not os.path.exists(rl_agents_folder):
            os.mkdir(rl_agents_folder)
        for i in range(len(self.rl_agents)):
            rl_ag_fol = os.path.join(rl_agents_folder, str(i))
            if not os.path.exists(rl_ag_fol):
                os.mkdir(rl_ag_fol)
            self.rl_agents[i].save_info(rl_ag_fol)


        pop_folder = os.path.join(folder_path, 'pop')
        if not os.path.exists(pop_folder):
            os.mkdir(pop_folder)
        for i in range(len(self.pop)):
            gene_ag_fol = os.path.join(pop_folder, str(i))
            if not os.path.exists(gene_ag_fol):
                os.mkdir(gene_ag_fol)
            self.pop[i].save_info(gene_ag_fol)
        
        with open(os.path.join(folder_path, 'count_actors.pkl'), 'wb') as f:
            pickle.dump(self.args.count_actors, f)
            print("Saved count: ", self.args.count_actors)

        self.archive.save_info()
        
    
    def load_info_mo(self, folder_path):
        rl_agents_folder = os.path.join(folder_path, "rl_agents")
        for i in range(len(self.rl_agents)):
            rl_ag_fol = os.path.join(rl_agents_folder, str(i))
            self.rl_agents[i].load_info(rl_ag_fol)


        pop_folder = os.path.join(folder_path, 'pop')
        ##########################################
        num_actors = len(os.listdir(pop_folder))
        ##########################################
        for i in range(num_actors):
            gene_ag_fol = os.path.join(pop_folder, str(i))
            new_genetic_agent = ddpg.GeneticAgent(self.args)
            self.pop.append(new_genetic_agent)
            self.pop[i].load_info(gene_ag_fol)
        
        with open(os.path.join(folder_path, 'count_actors.pkl'), 'rb') as f:
            self.args.count_actors = pickle.load(f)
            print("Loaded count: ", self.args.count_actors)

        self.archive.load_info()
    
    def save_info_warm_up(self, folder_path):        
        rl_agents_folder = os.path.join(folder_path, "rl_agents")
        if not os.path.exists(rl_agents_folder):
            os.mkdir(rl_agents_folder)
        for i in range(len(self.rl_agents)):
            rl_ag_fol = os.path.join(rl_agents_folder, str(i))
            if not os.path.exists(rl_ag_fol):
                os.mkdir(rl_ag_fol)
            self.rl_agents[i].save_info(rl_ag_fol)

        for rl_id in range(len(self.rl_agents)):
            pop_folder = os.path.join(folder_path, 'pop' + str(rl_id))
            if not os.path.exists(pop_folder):
                os.mkdir(pop_folder)
            for i in range(len(self.pop_list[rl_id])):
                gene_ag_fol = os.path.join(pop_folder, str(i))
                if not os.path.exists(gene_ag_fol):
                    os.mkdir(gene_ag_fol)
                self.pop_list[rl_id][i].save_info(gene_ag_fol)

    def load_info_warm_up(self, folder_path):
        rl_agents_folder = os.path.join(folder_path, "rl_agents")
        for i in range(len(self.rl_agents)):
            rl_ag_fol = os.path.join(rl_agents_folder, str(i))
            self.rl_agents[i].load_info(rl_ag_fol)


        for rl_id in range(len(self.rl_agents)):
            pop_folder = os.path.join(folder_path, 'pop' + str(rl_id))
            for i in range(len(self.pop_list[rl_id])):
                gene_ag_fol = os.path.join(pop_folder, str(i))
                self.pop_list[rl_id][i].load_info(gene_ag_fol)

    
    def save_info(self, checkpoint_folder: Optional[str] = None, warm_up_override: Optional[bool] = None):
        """Save a checkpoint.

        Args:
            checkpoint_folder: where to write the checkpoint. If None, uses self.checkpoint_folder.
            warm_up_override: if set, forces saving the warm-up (True) or stage-2 (False)
                checkpoint structure regardless of self.warm_up.
        """
        print("Saving info ......")
        folder_path = checkpoint_folder or self.checkpoint_folder
        os.makedirs(folder_path, exist_ok=True)

        warm_up_state = self.warm_up if warm_up_override is None else warm_up_override

        # Record phase explicitly so loading does not have to guess based on frame count.
        try:
            with open(os.path.join(folder_path, "phase.txt"), "w") as f:
                f.write("warm_up" if warm_up_state else "stage2")
        except OSError:
            pass

        info = os.path.join(folder_path, "info.npy")
        with open(info, "wb") as f:
            np.save(f, self.num_frames)
            np.save(f, self.gen_frames)
            np.save(f, self.num_games)
            np.save(f, self.trained_frames)
            np.save(f, self.iterations)
            if warm_up_state:
                np.save(f, np.array(self.fitness_list))
            else:
                np.save(f, self.fitness)
                np.save(f, np.array(self.pop_individual_type))

        if warm_up_state:
            wa_folder_path = os.path.join(folder_path, "warm_up")
            if not os.path.exists(wa_folder_path):
                os.mkdir(wa_folder_path)
            self.save_info_warm_up(wa_folder_path)
        else:
            mo_folder_path = os.path.join(folder_path, "multiobjective")
            if not os.path.exists(mo_folder_path):
                os.mkdir(mo_folder_path)
            self.save_info_mo(mo_folder_path)
        print("Saving checkpoint done!")
    
    def load_info(self, checkpoint_folder: Optional[str] = None):
        folder_path = checkpoint_folder or self.checkpoint_folder
        info = os.path.join(folder_path, "info.npy")
        with open(info, "rb") as f:
            self.num_frames = np.load(f)
            print("Num frames: ", self.num_frames)
            self.gen_frames = np.load(f)
            self.num_games = np.load(f)
            self.trained_frames = np.load(f)
            self.iterations = np.load(f)

            # 1. Prefer explicit phase metadata if present
            phase_file = os.path.join(folder_path, "phase.txt")
            phase = None
            if os.path.exists(phase_file):
                try:
                    with open(phase_file, "r") as pf:
                        phase = pf.read().strip()
                except OSError:
                    phase = None

            if phase == "warm_up":
                self.warm_up = True
            elif phase == "stage2":
                self.warm_up = False
            else:
                # 2. Backward-compatible heuristic: guess state based on frames
                # If frames > warm_up_frames, it sets self.warm_up = False
                if np.sum((self.num_frames <= self.args.warm_up_frames).astype(np.int32)) == 0:
                    print(self.num_frames)
                    self.warm_up = False
            
            # 2. Robust Loading Logic
            if self.warm_up:
                # If we explicitly know it's warm_up, load the list
                self.fitness_list = np.load(f)
            else:
                # We THINK it is MO mode, but the file might be from Warm-Up.
                try:
                    # Try to load the MO fitness array
                    self.fitness = np.load(f)
                    # Try to load the individual types (this fails if it's a warm-up save)
                    self.pop_individual_type = list(np.load(f))
                except EOFError:
                    print("Warning: Frame count indicates MO phase, but file structure indicates Warm-up phase.")
                    print("Reverting to Warm-up state to trigger natural transition.")
                    
                    # The data we just loaded into 'self.fitness' was actually 'fitness_list'
                    # So we move it to the correct variable.
                    self.fitness_list = self.fitness
                    
                    # Force warm_up back to True. 
                    # The main training loop will see frames > limit and trigger flatten_list() naturally.
                    self.warm_up = True
                    
                    # Reset self.fitness to a clean empty state so we don't leave garbage data
                    self.fitness = np.zeros((self.args.pop_size, self.num_objectives))

        # 3. Load the sub-folders based on the (possibly corrected) warm_up flag
        if self.warm_up:
            wa_folder_path = os.path.join(folder_path, "warm_up")
            self.load_info_warm_up(wa_folder_path)
        else:
            mo_folder_path = os.path.join(folder_path, "multiobjective")
            self.load_info_mo(mo_folder_path)

    def save_warm_up_info_file(self, logger, checkpoint_folder: Optional[str] = None):
        folder_path = checkpoint_folder or self.checkpoint_folder
        info = os.path.join(folder_path, "info.npy")
        wu_info = os.path.join(folder_path, "wu_info.npy")
        shutil.copy(info, wu_info)
        logger.info("=>>>>>> Saving warmup info successfully!")


