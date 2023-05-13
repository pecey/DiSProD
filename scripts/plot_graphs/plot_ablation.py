import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from scipy.io import loadmat

plt.rcParams["image.origin"] = "lower"
plt.rcParams["image.cmap"] = "jet"
plt.rcParams["image.interpolation"] = "gaussian"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8

master_mapping = {"cartpole": {"env_name": "continuous-cartpole-v1",
                                "title": "Continuous Cart Pole",
                                "alphas": ['0.0', '0.5', '0.75', '1', '2', '5', '7.5', '10']
                                },
                "mountain_car": {"env_name": "continuous-mountain-car-v2",
                                "title": "Continuous Mountain Car",
                                "alphas": ['0.0', '0.0025', '0.005', '0.0075', '0.009', '0.01', '0.02', '0.05']
                                },
                "pendulum":     {"env_name": "pendulum-v2",
                                "title": "Pendulum",
                                "alphas":['0.0', '0.25', '0.5', '0.75', '1', '1.25', '1.5', '1.75', '2.0']
                                },
		"dubins_car": {"env_name": "continuous-dubins-car-ablation-v0",
                                "title": "Continuous Dubins Car",
                                "alphas":['0','0.001', '0.005', '0.01', '0.05', '0.1']
                                },
                "simple_env": {"env_name": "simple-env-v1",
                               "title": "Simple Env",
                                "alphas": ['0', '0.1', '0.25', '0.5', '0.75', '1', '1.25', '1.5', '1.75', '2', '2.25', '2.5']
                                }
                }

mode_mapping = {
    "complete": ["Mean + AV + SV", "#2ca02c"],
    "state_var_only": ["Mean + SV", "#1f77b4"],
    "no_var": ["Mean", "#ff7f0e"]
}

DISPROD_PATH = os.getenv("DISPROD_PATH")
DISPROD_RESULTS_PATH = os.path.join(DISPROD_PATH, "results")

def main():
    run_name = "01-18-2023-ablation"
    env =  "cartpole"
    env_name = master_mapping[env]["env_name"]
    title = master_mapping[env]["title"]
    alphas = master_mapping[env]["alphas"]

    modes=['complete', 'state_var_only', 'no_var']

    run_base_path = f"{DISPROD_RESULTS_PATH}/results/{env_name}/planning/{run_name}"

    statistics = {}
    statistics_grouped = {}

    results_base_path = f"{run_base_path}"

    for mode in modes:
        statistics[mode] = {}
        statistics_grouped[mode] = {}
        for alpha in alphas:
            statistics[mode][alpha] = {}
            path = f"{results_base_path}/{run_name}-{mode}-{alpha}/logs/summary.log"
            with open(path, 'r') as f:
                data = f.readlines()
            mean, sd = [float(el.split(":")[1].strip()) for el in data[-1].strip("\n").split(",")]
            statistics[mode][alpha] = {"mean": mean, "sd": sd}

             # Divide rewards into groups and compute mean of means and sd of means.
            path_to_rewards = f"{results_base_path}/{run_name}-{mode}-{alpha}/logs/rewards.mat"
            data = loadmat(path_to_rewards)["rewards"]
            sorted_data = data[data[:, 0].argsort()]
            
            reward_groups = np.split(sorted_data, 8)
            mean_groups = [np.mean(g[:, 1]) for g in reward_groups]
            statistics_grouped[mode][alpha] = {"mean": np.mean(mean_groups), "sd": np.std(mean_groups)}


    with open(f"{results_base_path}/graph_summary.txt", "w") as f:
        f.write(json.dumps(statistics))


    figure = plt.figure()
    x = np.array(alphas).astype('float')
    for mode in modes:
        mean = []
        sd = []
        for alpha in alphas:
            mean.append(statistics_grouped[mode][alpha]["mean"])
            sd.append(statistics_grouped[mode][alpha]["sd"])
        mean = np.array(mean)
        sd = np.array(sd)
        plt.plot(x, mean, label=mode_mapping[mode][0], color=mode_mapping[mode][1])
        plt.fill_between(x, mean-sd, mean+sd, alpha=0.2, color=mode_mapping[mode][1])
    plt.grid("on")
    plt.autoscale(tight=True)
    plt.xlabel(f"Alpha")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(f"{run_base_path}/ablation_{env}.pdf", format='pdf', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()

