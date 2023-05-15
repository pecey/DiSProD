# DiSProD: Differentiable Symbolic Propagation of Distributions for Planning 

[DiSProD](planners/continuous_disprod.py) is an online planner developed for environments with probabilistic transitions with continuous action spaces. DiSProD builds a symbolic graph that captures the distribution of future trajectories, conditioned on a given policy, using independence assumptions and approximate propagation of distributions. The symbolic graph provides a differentiable representation of the policy’s value, enabling efficient gradient-based optimization for long-horizon search. The propagation of approximate distributions can be seen as an aggregation of many trajectories, making it well-suited for dealing with sparse rewards and stochastic environments. While most of our experiments are with continuous state spaces, we can work with hybrid state spaces as well. We also provide [an implementation](planners/discrete_disprod.py) for a version of DiSProD that works with binary action spaces, but it has not been tested extensively.

## Instructions

#### Setting up repo
```
git clone git@github.com:pecey/DiSProD.git
cd DiSProD
```
#### Installing dependencies

```py
pip install -r requirements.txt
```

#### Instructions to run

We assume that an environment variable called `DISPROD_PATH` is set and points to this folder. Please set that before running any of the scripts.

To run the planner on an environment:
```py
python run_gym.py --alg=<alg_name> --env=<env_code>
```
`alg_name` can be one of the following: `disprod`, `cem`, `mppi`, while `env_code` can be one of the following codes. As shooting algorithms are configured for discrete environments, running `cem` or `mppi` for (Discrete) Cartpole `cp` or (Discrete) Mountain Car `mc` will raise an Exception.

| Environment                               | env_code        |
| -----------                               | -----------     |
| CartPole                                  | `cp`            |
| Mountain Car                              | `mc`            |
| Continuous CartPole                       | `ccp`           |
| Continuous Mountain Car                   | `cmc`           |
| Pendulum                                  | `p`             |
| Continuous Dubins Car                     | `cdc`           |
| Continuous Mountain Car - Sparse Rewards  | `cmc_sp`        |
| Continuous Mountain Car - High Dimension  | `cmc_hd`        |
| Continuous CartPole - Hybrid              | `ccp_h`         |

The other configurations available via CLI are:
- `--seed`: Specify base seed to PRNG.
- `--depth`: Planning horizon for the planner.
- `--alpha`: Controls noise levels in the environment. For no noise, use `--alpha=0`.
- `--reward_sparsity`: Controls the sparsity of rewards in the planner. Higher the value, sparser the reward.
- `--n_episodes`: Number of episodes to run the planner for.
- `--map_name`: Specify the map name to use for Dubins Car environments. Valid values are listed below
- `--n_samples`: Specify the population size for CEM/MPPI.
- `--step_size`: Specify the step size for action mean updates in DiSProD. 
- `--step_size_var`: Specify the step size for action variance updates in DiSProD.
- `--n_restarts`: Specify the number of restarts in DiSProD.
- `--taylor_expansion_mode`: Control which mode of DiSProD to use. Valid values are `complete`, `state_var` and `no_var`.
- `--n_actions`: Specify the number of actions for Continuous Mountain Car - High Dimension.  
- `--render`: Control whether to render the Gym interactions.

###### Maps for Dubins Car environments

All the map configurations are defined in [env/assets/dubins.json](env/assets/dubins.json). At present, the available maps are `no-ob-[1-5]`, `ob-[1-11]`, `u` and `cave-mini`.

#### SLURM scripts

The scripts for running the experiments are in `slurm_scripts` and are geared towards clusters being managed by SLURM. The folder has bash scripts for all the experiments which in-turn submit the respective jobs to SLURM. The scripts take the name of the environment as arguments. Valid values are `cartpole`, `pendulum`, `mountain_car`, `dubins_car` and `simple_env`. 

For example, to trigger the experiments that vary the noise levels on mountain car environment, 

```sh
./trigger_exp_noise.sh mountain_car
```

The equivalent `python` command for running the same experiment is:

```py
PYTHONPATH=. python run_gym.py --env=cmc --alg=${alg} --n_episodes=48 --alpha=${alpha_val} --render=False 
```

where `alg` is one of `disprod`, `cem` or `mppi` and `alpha_val` controls the amount of noise.

#### Code Structure

- `config` contains domain-specific configuration used by the planners. `config/default.yaml` contains the default config which can be overwritten by using the same key in the domain-specific file.
- `env` contains the environment files for each of the domains. `env/transition_fns.py` and `env/reward_fns.py` contains the transition functions and the reward functions for each of the domains. These functions are provided to the planner via the config files. 
- `planners` contains the planning algorithms. `continuous_disprod` is for continuous action spaces and can handle both continuous and binary state variables. `discrete_disprod` is for binary action spaces. `planners/baseline` contains the code for CEM and MPPI.
- `utils` contains helper methods that are used in various places.

#### Adding a new environment

Using DiSProD on a new environment is straightforward. 
- Have a JAX-compatible transition function with all noises represented as inputs.
- Have a JAX-compatible reward function.
- Since we register the environments ourselves with Gym, a copy of the environment file would also be required. This step can be skipped but would require manually commenting out portions of the code.
- Add a configuration file in `config` with atleast the following keys:
      - `env_name`: The name to register with Gym.
      - `env_entry_point`: Path to the simulator file.
      - `max_episode_steps`: Max episodes steps permissible.
      - `reward_fn`: Path to the JAX-compatible reward function.
      - `transition_fn`: Path to the JAX-compatible transition function.
      - `discrete`: True if the action space is discrete. This is used to call the relevant version of DiSProD while planning. 
- Add a entry in `ENV_MAPPING` in `run_gym.py` mapping the environment code-name that you want with the name of the config file.
- Add the environment code name as a valid choice for `--env` argument for Argparse in `run_gym.py`


#### Experiments with ROS

- Instructions for experiments with TurtleBot can be found in [ros1-turtlebot/README.md](ros1-turtlebot/README.md)

#### Debugging

As the main loop to run experiments uses multiprocessing, if the inner loop fails, the outer loop still tries to execute and errors out with a dimensionality issue for the reward matrix. The error message looks something like:

```sh
Traceback (most recent call last):
  File "run_gym.py", line 171, in <module>
    main(args)
  File "run_gym.py", line 122, in main
    rewards = rewards_matrix[:, 1]
IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
```

This message can be ignored as it is not the true source of error. The actual error would be found above this message in the stacktrace.

## Bibtex
If you find this work useful, please cite

```bibtex
@misc{chatterjee2023disprod,
      title={DiSProD: Differentiable Symbolic Propagation of Distributions for Planning}, 
      author={Palash Chatterjee and Ashutosh Chapagain and Weizhe Chen and Roni Khardon},
      year={2023},
      eprint={2302.01491},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```