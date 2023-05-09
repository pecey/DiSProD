import gym
import numpy as onp
import torch as T
from matplotlib import pyplot as plt
from matplotlib import animation
import importlib
import random
import jax.numpy as jnp
from jax import random as jax_random
from omegaconf import OmegaConf
import os
from datetime import date
import time
from pathlib import Path


# Argmax with random tie-breaks
# The 0th-restart is always set to the previous solution
def random_argmax(key, x, pref_idx=0):
    try:
        options = jnp.where(x == jnp.nanmax(x))[0]
        val = 0 if 0 in options else jax_random.choice(key, options)
    except:
        val = jax_random.choice(key, jnp.arange(len(x)))
        print(f"All restarts where NaNs. Randomly choosing {val}.")
    finally:
        return val


# Miscellaneous functions
# Dynamically load a function from a module
def load_method(name):
    mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn

# Print to file or stdout
def print_(x, path=None, flush=True):
    if path is None:
        print(x)
    else:
        print(x, file=open(path, 'a'), flush=flush)


# https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553
def save_frames_as_gif(frames, path='./', filename='gym_animation.gif', save_as_video=False):
    # Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(f"{path}/{filename}", writer='imagemagick', fps=60)
    if save_as_video:
        video_filename = filename.replace(".gif", ".mp4")
        anim.save(f"{path}/{video_filename}", extra_args=['-vcodec', 'libx264'])
    plt.close()
    print(f"Saving GIF in {path}/{filename}, frames : {len(frames)}")
    # plt.savefig(f"{path}{filename}_{idx}.png")


# Seed the libraries
def set_global_seeds(seed, env=None):
    onp.random.seed(seed)
    random.seed(seed)

    T.manual_seed(seed)
    if T.cuda.is_available():
        T.cuda.manual_seed_all(seed)
        T.backends.cudnn.deterministic = True
        T.backends.cudnn.benchmark = False

    if env:
        env.seed(seed)
        env.action_space.seed(seed)


# Register environment
def setup_environment(config):
    from gym.envs.registration import register

    # TODO: A better way to do this has been introduced in Python 3.9. Using dict_a | dict_b
    kwargs_dict = {'alpha': config['alpha']}
    if "dubins" in config['env_name']:
        kwargs_dict.update({'add_obstacle': config['add_obstacles'],
                            'config_file': config['obstacles_config_path'],
                            'map_name': config['map_name']}),

    if  config["env_name"] in ["sparse-continuous-mountain-car-v1", "simple-env-v1"]:
        kwargs_dict.update({'sparsity': config['reward_sparsity']})
        
    if config["env_name"] in ["continuous-mountain-car-v3"]:
        kwargs_dict.update({'n_actions': config['n_actions']})
        
    # if config["env_name"] in ["continuous-cartpole-hybrid-v1"]:
    #    kwargs_dict.update({'ignore_shaky_in_planner': config["ignore_shaky_in_planner"]})
        
    register(
        id=config['env_name'],
        entry_point=config['env_entry_point'],
        max_episode_steps=config['max_episode_steps'],
        kwargs=kwargs_dict
    )
    env = gym.make(config['env_name'])
    return env

def load_config_if_exists(path, log_path):
    if os.path.exists(path):
        print_(f"Using config file : {path}", log_path)
        return OmegaConf.load(path)
    print_(f"Requested config file not found at : {path}. Skipping", log_path)
    return {}

def prepare_config(env_name, mode, cfg_path=None, log_path=None):
    if mode not in ["planning"]:
        raise Exception(f"Unable to parse config files. Unknown mode passed. Mode received: {mode}, but was expecting planning")

    config_files = [f"{cfg_path}/{mode}/default.yaml",
                    f"{cfg_path}/{mode}/{env_name}.yaml",
                    f"{cfg_path}/disprod_default.yaml",
                    f"{cfg_path}/default.yaml",
                    f"{cfg_path}/{env_name}.yaml"]
    config = {}
    for config_file in config_files:
        config = OmegaConf.merge(config, load_config_if_exists(config_file, log_path))
    return config

def update_config_with_args(env_cfg, args, base_path):
    # Non-boolean keys
    keys_to_update = ["seed", "log_file", "depth", "n_episodes", "alg", "alpha", "reward_sparsity", "taylor_expansion_mode"]
    for key in keys_to_update:
        if args.__contains__(key) and getattr(args,key) is not None:
            env_cfg[key] = getattr(args, key)
            
    if args.__contains__("n_restarts") and getattr(args,"n_restarts") is not None:
        env_cfg["disprod"]["n_restarts"] = getattr(args, "n_restarts")
        
    if args.__contains__("taylor_expansion_mode") and getattr(args,"taylor_expansion_mode") is not None:
        env_cfg["disprod"]["taylor_expansion_mode"] = getattr(args, "taylor_expansion_mode")
         
    if args.__contains__("step_size") and getattr(args,"step_size") is not None:
        env_cfg["disprod"]["step_size"] = getattr(args, "step_size")
    
    if args.__contains__("n_samples") and getattr(args,"n_samples") is not None:
        if env_cfg["alg"] in ["cem", "hybrid_cem"]:
            env_cfg["cem"]["n_samples"] = getattr(args, "n_samples")
        elif env_cfg["alg"] in ["mppi", "hybrid_mppi"]:
            env_cfg["mppi"]["n_samples"] = getattr(args, "n_samples")
        else:
            raise Exception(f"Cannot set n_samples for alg {env_cfg['alg']}")

    
    if args.env_name == "continuous_mountain_car_high_dim":
        if args.__contains__("n_actions") and getattr(args,"n_actions") is not None:
            env_cfg["n_actions"] = getattr(args, "n_actions")
        
    # Boolean keys
    boolean_keys = ["render", "headless"]
    for key in boolean_keys:
        if args.__contains__(key) and getattr(args,key) is not None:
            env_cfg[key] = getattr(args, key).lower() == "true"

    # If run_name is set, the update in config. Else set default value to {running_mode}_{current_time}
    if args.__contains__("run_name") and getattr(args, "run_name"):
        env_cfg["run_name"] = args.run_name
    else:
        today = date.today()
        env_cfg["run_name"] = f"{today.strftime('%y-%m-%d')}_{int(time.time())}"

    # Map for dubins        
    if "dubins" in env_cfg["env_name"]:
        if getattr(args, "obstacles_config_file").lower() == "none":
            env_cfg["obstacles_config_path"] = None
        else:
            env_cfg["obstacles_config_path"] = f"{base_path}/env/assets/{args.obstacles_config_file}.json"
            env_cfg["map_name"] = args.map_name
            
    return env_cfg

def setup_output_dirs(cfg, run_name, base_path, setup_for_learning):
    if setup_for_learning:
        base_dir = f"{base_path}/results/{cfg['env_name']}/learning/{run_name}"
    else:
        base_dir = f"{base_path}/results/{cfg['env_name']}/planning/{run_name}"
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    cfg["results_dir"] = base_dir

    log_dir = f"{base_dir}/logs"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    cfg["log_dir"] = log_dir
    cfg["log_file"] = f"{log_dir}/debug.log"

    graph_dir = f"{base_dir}/graphs"
    Path(graph_dir).mkdir(parents=True, exist_ok=True)
    cfg["graph_dir"] = graph_dir

    if setup_for_learning:
        model_dir = f"{base_dir}/model"
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        cfg["model_dir"] = model_dir

        data_dir = f"{base_dir}/data"
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        cfg["data_dir"] = data_dir
