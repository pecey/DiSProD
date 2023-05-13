import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import json
import glob
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
                                "alphas": [0.0, 0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                                },
                "mountain_car": {"env_name": "continuous-mountain-car-v2",
                                "title": "Continuous Mountain Car",
                                "alphas": [0.0, 0.0025, 0.005, 0.0075, 0.009, 0.01, 0.02, 0.05]
                                },
                "pendulum":     {"env_name": "pendulum-v2",
                                "title": "Pendulum",
                                "alphas":[0.0, 0.25, 0.5, 0.75, 1, 2, 3, 4, 5]
                                },
                }

alg_mapping = {
    "cem": "CEM",
    "mppi": "MPPI",
     "disprod": "DiSProD"
}

DISPROD_PATH = os.getenv("DISPROD_PATH")
DISPROD_RESULTS_PATH = os.path.join(DISPROD_PATH, "results")


def main():
    run_name = "10-05-2022-exp-noise"
    env = "cartpole"

    env_name = master_mapping[env]["env_name"]
    title = master_mapping[env]["title"]
    alphas = master_mapping[env]["alphas"]

    algorithms =["cem", "mppi", "disprod"]

    run_base_path = f"{DISPROD_RESULTS_PATH}/{env_name}/planning/{run_name}"

    statistics = {}
    statistics_grouped = {}

    results_base_path = f"{run_base_path}"

    print(f"Graph location: {results_base_path}/exp_noise_{env}.pdf")

    for algorithm in algorithms:
        statistics[algorithm] = {}
        statistics_grouped[algorithm] = {}
        for alpha in alphas:
            statistics[algorithm][alpha] = {}
            path = f"{results_base_path}/{run_name}-{algorithm}-{alpha}/logs/summary.log"
            with open(path, 'r') as f:
                data = f.readlines()
            mean, sd = [float(el.split(":")[1].strip()) for el in data[-1].strip("\n").split(",")]
            statistics[algorithm][alpha] = {"mean": mean, "sd": sd}

            # Divide rewards into groups and compute mean of means and sd of means.
            path_to_rewards = f"{results_base_path}/{run_name}-{algorithm}-{alpha}/logs/rewards.mat"
            data = loadmat(path_to_rewards)["rewards"]
            sorted_data = data[data[:, 0].argsort()]
            
            reward_groups = np.split(sorted_data, 8)
            mean_groups = [np.mean(g[:, 1]) for g in reward_groups]
            statistics_grouped[algorithm][alpha] = {"mean": np.mean(mean_groups), "sd": np.std(mean_groups)}


    with open(f"{run_base_path}/graph_summary.txt", "w") as f:
        f.write(json.dumps(statistics))


    x = np.array(alphas)

    figure = plt.figure()
    for algorithm in algorithms:
        mean = []
        sd = []
        for alpha in alphas:
            mean.append(statistics[algorithm][alpha]["mean"])
            sd.append(statistics[algorithm][alpha]["sd"])
        mean = np.array(mean)
        sd = np.array(sd)
        plt.plot(x, mean, label=alg_mapping[algorithm])
        plt.fill_between(x, mean-sd, mean+sd, alpha=0.2)
    plt.legend()
    plt.grid("on")
    plt.autoscale(tight=True)
    plt.xlabel(f"Alpha")
    plt.ylabel("Score")
    plt.title(f"Env: {title}")
    plt.tight_layout()
    plt.savefig(f"{run_base_path}/exp_noise_{env}.pdf", format='pdf', bbox_inches='tight')
    plt.close()

    figure = plt.figure()
    for algorithm in algorithms:
        mean = []
        sd = []
        for alpha in alphas:
            mean.append(statistics_grouped[algorithm][alpha]["mean"])
            sd.append(statistics_grouped[algorithm][alpha]["sd"])
        mean = np.array(mean)
        sd = np.array(sd)
        plt.plot(x, mean, label=alg_mapping[algorithm])
        plt.fill_between(x, mean-sd, mean+sd, alpha=0.2)
    plt.legend()
    plt.grid("on")
    plt.autoscale(tight=True)
    plt.xlabel(f"Alpha")
    plt.ylabel("Score")
    plt.title(f"Env: {title}")
    plt.tight_layout()
    plt.savefig(f"{run_base_path}/exp_noise_{env}_grouped.pdf", format='pdf', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
