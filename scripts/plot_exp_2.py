import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import json
import glob

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

env_name_mapping = {
    "continuous-cartpole-v1": "Continuous Cart Pole",
    "continuous-mountain-car-v2": "Continuous Mountain Car",
    "pendulum-v2": "Pendulum"
}

alg_mapping = {
    "cem": "CEM",
    "mppi": "MPPI",
     "disprod": "DiSProD"
}

DISPROD_PATH = os.getenv("DISPROD_PATH")
DISPROD_RESULTS_PATH = os.path.join(DISPROD_PATH, "results")


def main():
    run_name = "06-13-2022-nsamples-vs-performance-stochastic"
    env_name = "pendulum-v2"
    folder_name = "06-13-2022-exp2"

    algorithms =["cem", "mppi", "sogbofa"]
    n_samples=[10, 20, 30, 40, 50, 100, 150, 200, 350, 500]
    # n_samples = [10, 25, 50, 75, 100, 150, 250, 500, 750, 1000, 1500, 2000]


    run_base_path = f"{DISPROD_RESULTS_PATH}/{env_name}/{run_name}"
    env = env_name_mapping[env_name]

    statistics = {}
    for algorithm in algorithms:
        statistics[algorithm] = {}
        mean_list = []
        sd_list = []
        for n_sample in n_samples:
            statistics[algorithm][n_sample] = {}
            filename = glob.glob(f"{run_base_path}/data/{folder_name}-{algorithm}_{n_sample}/results*.txt")[0]
            with open(f"{filename}", 'r') as f:
                data = f.readlines()
            mean, sd = [float(el.split(" ")[-1]) for el in data[-1].strip("\n").strip().split(",")]
            mean_list.append(mean)
            sd_list.append(sd)
        statistics[algorithm] = {"mean": mean_list, "sd": sd_list}


    with open(f"{run_base_path}/summary.txt", "w") as f:
        f.write(json.dumps(statistics))


    x = np.array(n_samples)
    for algorithm in algorithms:
        mean = np.array(statistics[algorithm]["mean"])
        sd = np.array(statistics[algorithm]["sd"])
        plt.plot(x, mean, label=alg_mapping[algorithm])
        plt.fill_between(x, mean-sd, mean+sd, alpha=0.2)
    plt.legend()
    plt.grid("on")
    plt.autoscale(tight=True)
    plt.xlabel(f"Number of samples/restarts")
    plt.ylabel("Score")
    plt.title(f"Env: Noisy {env} - (Alpha = 2)")
    plt.tight_layout()
    plt.savefig(f"{run_base_path}/exp2_noisy.pdf", format='pdf', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()