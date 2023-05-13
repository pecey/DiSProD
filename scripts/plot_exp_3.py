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
    run_name = "06-14-2022-noise-vs-performance-2000"
    env_name = "continuous-mountain-car-v2"
    folder_name = "06-14-2022-exp3-2000-restarts"

    algorithms =["cem", "mppi", "sogbofa"]
    alphas=['0.0', '0.0025', '0.005', '0.0075', '0.009', '0.01', '0.02', '0.05']


    run_base_path = f"{DISPROD_RESULTS_PATH}/{env_name}/{run_name}"
    env = env_name_mapping[env_name]

    statistics = {}
    for algorithm in algorithms:
        statistics[algorithm] = {}
        mean_list = []
        sd_list = []
        for alpha in alphas:
            statistics[algorithm][alpha] = {}
            filename = glob.glob(f"{run_base_path}/data/{folder_name}-{algorithm}_{alpha}/results*.txt")[0]
            with open(f"{filename}", 'r') as f:
                data = f.readlines()
            mean, sd = [float(el.split(" ")[-1]) for el in data[-1].strip("\n").strip().split(",")]
            mean_list.append(mean)
            sd_list.append(sd)
        statistics[algorithm] = {"mean": mean_list, "sd": sd_list}


    with open(f"{run_base_path}/summary.txt", "w") as f:
        f.write(json.dumps(statistics))


    x = np.array(alphas)
    for algorithm in algorithms:
        mean = np.array(statistics[algorithm]["mean"])
        sd = np.array(statistics[algorithm]["sd"])
        plt.plot(x, mean, label=alg_mapping[algorithm])
        plt.fill_between(x, mean-sd, mean+sd, alpha=0.2)
    plt.legend()
    plt.grid("on")
    plt.autoscale(tight=True)
    plt.xlabel(f"Alpha")
    plt.ylabel("Score")
    plt.title(f"Env: {env}")
    plt.tight_layout()
    plt.savefig(f"{run_base_path}/exp3.pdf", format='pdf', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()