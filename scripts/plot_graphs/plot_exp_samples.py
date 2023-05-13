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
                                "title": "Hybrid Continuous Cart Pole",
                                "n_samples": [10, 20, 30, 40, 50, 100, 150, 200, 350, 500]
                                },
                "mountain_car": {"env_name": "continuous-mountain-car-v2",
                                "title": "Continuous Mountain Car",
                                "n_samples": [10, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
                                },
                "pendulum":     {"env_name": "pendulum-v2",
                                "title": "Pendulum",
                                "n_samples":[10, 20, 30, 40, 50, 100, 150, 200, 350, 500]
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
    run_name = "01-03-2023-nsamples-vs-performance"
    env = "cartpole"

    noisy = False
	

    if noisy:
        run_name = f"{run_name}-noisy"
        if env == "pendulum":
            alpha_val = 2
        elif env == "mountain_car":
            alpha_val = 0.002
        elif env == "cartpole":
            alpha_val = 5
    else:
        alpha_val = 0

    env_name = master_mapping[env]["env_name"]
    title = master_mapping[env]["title"] if not noisy else f'Noisy {master_mapping[env]["title"]} -  (Alpha = {alpha_val})'
    plot_file_name =  f"exp_samples" if not noisy else f"exp_samples_noisy"
    n_samples = master_mapping[env]["n_samples"]

    algorithms =["cem", "disprod", "mppi"]

    run_base_path = f"{DISPROD_RESULTS_PATH}/{env_name}/planning"

    statistics = {}
    statistics_grouped = {}

    results_base_path = f"{run_base_path}"

    print(f"Graph location: {results_base_path}/{plot_file_name}_{env}.pdf")

    for algorithm in algorithms:
        statistics[algorithm] = {}
        statistics_grouped[algorithm] = {}
        for n_sample in n_samples:
            statistics[algorithm][n_sample] = {}
            path = f"{results_base_path}/{run_name}-{algorithm}-{n_sample}/logs/summary.log"
            with open(path, 'r') as f:
                data = f.readlines()
            mean, sd = [float(el.split(":")[1].strip()) for el in data[-1].strip("\n").split(",")]
            statistics[algorithm][n_sample] = {"mean": mean, "sd": sd}

            # Divide rewards into groups and compute mean of means and sd of means.
            path_to_rewards = f"{results_base_path}/{run_name}-{algorithm}-{n_sample}/logs/rewards.mat"
            data = loadmat(path_to_rewards)["rewards"]
            sorted_data = data[data[:, 0].argsort()]
            
            reward_groups = np.split(sorted_data, 8)
            mean_groups = [np.mean(g[:, 1]) for g in reward_groups]
            statistics_grouped[algorithm][n_sample] = {"mean": np.mean(mean_groups), "sd": np.std(mean_groups)}

    with open(f"{run_base_path}/graph_summary.txt", "w") as f:
        f.write(json.dumps(statistics))


    x = np.array(n_samples)

    figure = plt.figure()
    for algorithm in algorithms:
        mean = []
        sd = []
        for n_sample in n_samples:
            mean.append(statistics[algorithm][n_sample]["mean"])
            sd.append(statistics[algorithm][n_sample]["sd"])
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
    plt.savefig(f"{run_base_path}/{plot_file_name}_{env}.pdf", format='pdf', bbox_inches='tight')
    plt.close()

    figure = plt.figure()
    for algorithm in algorithms:
        mean = []
        sd = []
        for n_sample in n_samples:
            mean.append(statistics_grouped[algorithm][n_sample]["mean"])
            sd.append(statistics_grouped[algorithm][n_sample]["sd"])
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
    plt.savefig(f"{run_base_path}/{plot_file_name}_{env}_grouped.pdf", format='pdf', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
