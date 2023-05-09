import argparse
import sys
import multiprocessing as mp
import numpy as np
import json
import jax

from utils.common_utils import print_, save_frames_as_gif, setup_environment, set_global_seeds, prepare_config, update_config_with_args, setup_output_dirs
import time
import os
from planners.gym_interface import setup_planner
from omegaconf import OmegaConf
from scipy.io import savemat

DISPROD_PATH = os.getenv("DISPROD_PATH")
sys.path.append(DISPROD_PATH)
DISPROD_CONF_PATH = os.path.join(DISPROD_PATH, "config")

ENV_MAPPING = { "ccp"   : "continuous_cartpole",
                "cmc"   : "continuous_mountain_car", 
                "p"     : "pendulum",
                "ccp_h"  : "continuous_cartpole_hybrid", 
                "cmc_sp"  : "sparse_continuous_mountain_car", 
                "cdc"   : "continuous_dubins_car",
                "cmc_hd" : "continuous_mountain_car_high_dim"}

def run(cfg, queue, n_episodes, seeds):
    scores = []
    for idx in range(n_episodes):
        env = setup_environment(cfg)
        set_global_seeds(seed=seeds[idx], env=env)
        key = jax.random.PRNGKey(seeds[idx])
        agent = setup_planner(env, cfg, key)

        done = False
        total_reward = 0
        n_step = 0

        # Reset everything
        obs = env.reset()
        ac_seq, key = agent.reset(key)
        # agent.reset()
        frames = []
        while not done:
            # _, imagined_trajectory = agent.choose_action(agent, env, obs)
            action, ac_seq, key = agent.choose_action(obs, ac_seq, key)
            # action, _ = agent.choose_action(obs)
            n_step += 1
            print(f"Step: {n_step}, State: {obs}, Action: {action}")
            if cfg['debug_planner']:
                print_(f"Step: {n_step}, State: {obs}, Action: {action}", cfg['log_file'])
            # if cfg["plot_imagined_trajectory"]:
            #     env.set_imagined_trajectory_data(imagined_trajectory)
            if cfg['render']:
                env.render()
            if cfg['save_as_gif']:
                frames.append(env.render(mode="rgb_array"))

            obs, reward, done, _ = env.step(action)
            total_reward += reward
            
        # Save frames
        if len(frames) > 0:
            save_frames_as_gif(frames, path=f"{cfg['graph_dir']}", filename=f"seed_{seeds[idx]}.gif")

        # Convert to list if reward is ndarray. Required just for Pendulum.
        if type(total_reward).__module__ in ['numpy', "jaxlib.xla_extension"]:
            total_reward = total_reward.tolist()
        
        # Dump scores as JSON
        scores.append({"seed": seeds[idx], "returns": total_reward, "steps": n_step})
        if "dubins" in cfg["env_name"]:
            env.save_trajectory(f"{cfg['graph_dir']}", f"{seeds[idx]}_trajectory")
    queue.put(scores)


def main(args):
    args.env_name = ENV_MAPPING[args.env]
    cfg = prepare_config(args.env_name, "planning", DISPROD_CONF_PATH)
    cfg = update_config_with_args(cfg, args, base_path=DISPROD_PATH)
    run_name = cfg["run_name"]

    # Setup virtual display for server
    if args.headless.lower() == "true" and cfg["save_as_gif"]:
        setup_virtual_display()

    setup_output_dirs(cfg, run_name, DISPROD_PATH, setup_for_learning=False)

    base_seed = cfg["seed"]
    partitions = np.minimum(mp.cpu_count(), cfg["n_episodes"])
    episode_per_partition = int(np.ceil(cfg["n_episodes"]/partitions))

    print(f"Evaluating {args.env_name} using {partitions} CPUs for {cfg['n_episodes']} episodes. Config {OmegaConf.to_yaml(cfg)}")

    seeds = list(range(base_seed, base_seed * ((episode_per_partition * partitions) + 1), base_seed))

    queue = mp.Queue()   
    processes = []
    scores = []

    for idx in range(0, len(seeds), episode_per_partition):
        p = mp.Process(target=run, args=(cfg, queue, episode_per_partition, seeds[idx:idx+episode_per_partition]))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

    while (not queue.empty()):
        scores.append(queue.get())

    with open(f"{cfg['log_dir']}/output.log", 'w') as f:
        json.dump(scores, f)

    rewards_matrix = np.array([(score["seed"], score["returns"]) for el in scores for score in el])
        
    rewards = rewards_matrix[:, 1]
    rewards_mean = np.mean(rewards)
    rewards_sd = np.std(rewards)
    
    print(f"Mean: {rewards_mean}, SD: {rewards_sd} \n")

    with open(f"{cfg['log_dir']}/summary.log", 'w') as f:
        f.write(f"Config: {OmegaConf.to_yaml(cfg)} \n")
        f.write(f"Mean: {rewards_mean}, SD: {rewards_sd} \n")
        
    savemat(os.path.join(cfg['log_dir'], "rewards.mat"),{
            "rewards": rewards_matrix
        })


def setup_virtual_display():
    from pyvirtualdisplay import Display
    display = Display(visible=0, size=(1400, 900))
    display.start()

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, choices=["ccp", "cmc", "p", "ccp_h", "cmc_sp", "cdc", "cmc_hd"], required=True)
    parser.add_argument('--render', type=str, default="True")
    parser.add_argument('--seed', type=int, help='Seed for PRNG', default=42)
    parser.add_argument('--run_name', type=str, help='Run Name', default=str(int(time.time())))
    parser.add_argument('--depth', type=int, help='Specifies the planning horizon for the planner.')
    parser.add_argument('--alpha', type=float, help='Controls noise levels in the simulator.')
    parser.add_argument('--reward_sparsity', type=float, help='Controls the sparsity of rewards in the planner.')
    parser.add_argument('--n_episodes', type=int, help='Number of episodes to run the experiment for.')
    parser.add_argument('--alg', type=str, help='Specify which algorithm to use - DiSProD/MPPI/CEM/Hybrid DiSProD/Hybrid CEM', choices=['disprod', 'mppi', 'cem', 'hybrid_disprod' , 'hybrid_cem'])
    parser.add_argument('--obstacles_config_file', type=str, help="Config filename without the JSON extension",
                        default="dubins")
    parser.add_argument('--map_name', type=str, help="Specify the map name to be used. Only called if dubins or continuous dubins env", default="no-ob-1")
    parser.add_argument('--headless', type=str, help="If set to True, then the program is being run on server",
                        default="False")
    # CEM/MPPI
    parser.add_argument('--n_samples', type=int, help='Number of samples to sample in CEM/MPPI')
    # DiSProD specific
    parser.add_argument('--step_size', type=float, help='Controls the step-size in DiSProD in the planner')
    parser.add_argument('--taylor_expansion_mode', type=str, help="Control the use of variance in Taylor's expansion", choices=['complete', 'state_var_only', 'action_var_only', 'no_var'])
    parser.add_argument('--n_restarts', type=int, help='Number of restarts to perform in DiSProD')
    # For experiments with continuous-mountain-car-high-dim
    parser.add_argument('--n_actions', type=int, help="Varying the number of actions. n_actions = n_redundant_actions + 1", default=1)

    args, unknown = parser.parse_known_args()
    main(args)
