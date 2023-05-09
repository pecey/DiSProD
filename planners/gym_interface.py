from planners.discrete_disprod import DiscreteDisprod
# from planners.continuous_disprod import ContinuousDisprod
from planners.continuous_disprod_hybrid import ContinuousDisprodHybrid
from planners.c_disprod import ContinuousDisprod
from planners.baseline import shooting_cem , shooting_cem_hybrid, cem
import numpy as onp

# def plan_one_step(agent, env, state):
#     action, imagined_trajectory = agent.choose_action(state)
#     return onp.array(action), imagined_trajectory


def setup_planner(env, env_cfg, key):
    if env_cfg["alg"] in ["cem", "mppi"]:
        agent = cem.ShootingCEM(env, env_cfg)
    elif env_cfg["alg"] == "disprod":
        if env_cfg["discrete"]:
            agent = DiscreteDisprod(env, env_cfg)
        else:
            agent = ContinuousDisprod(env, env_cfg)
    elif env_cfg["alg"] == "hybrid_disprod":
        agent = ContinuousDisprodHybrid(env, env_cfg)
    elif env_cfg["alg"] == "hybrid_cem":
        agent = shooting_cem_hybrid.ShootingCEMHybrid(env, env_cfg)
    
    return agent
