def setup_planner(env, env_cfg, key):
    if env_cfg["alg"] in ["cem", "mppi"]:
        from planners.baseline.cem import ShootingCEM
        agent = ShootingCEM(env, env_cfg)
    elif env_cfg["alg"] == "disprod":
        if env_cfg["discrete"]:
            from planners.discrete_disprod import DiscreteDisprod
            agent = DiscreteDisprod(env, env_cfg)
        else:
            from planners.continuous_disprod import ContinuousDisprod
            agent = ContinuousDisprod(env, env_cfg)
    
    return agent
