# DiSProD: Differentiable Symbolic Propagation of Distributions for Planning 

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
`alg_name` can be one of the following: `disprod`, `cem`, `mppi`, `hybrid_disprod` , `hybrid_cem`, while `env_code` can be one of the following codes.

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
- `env` contains the environment files for each of the domains. `env/transition_fns.py` and `env/reward_fns.py` contains the transition functions and the reward functions for each of the domains. The definitions in these files are read by the planner.
- `planners` contains the planning algorithms. `planners/baseline` contains the code for CEM, MPPI and Hybrid-CEM.
- `utils` contains helper methods that are used in various places.


#### Experiments with ROS

- Instructions for experiments with TurtleBot can be found in [ros1-turtlebot/README.md](ros1-turtlebot/README.md)

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