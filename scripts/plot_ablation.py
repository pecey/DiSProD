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

mode_mapping = {
    "complete": "Mean + AV + SV",
    "state_var": "Mean + SV",
    "mean_prop": "Mean"
}

DISPROD_PATH = os.getenv("DISPROD_PATH")
DISPROD_RESULTS_PATH = os.path.join(DISPROD_PATH, "results")

def main():
    run_name = "06-14-2022-ablation"

    # env_name="pendulum-v2"
    # alphas=['0.0', '0.5', '0.75', '1', '2', '3', '4', '5']
    # env_name = "continuous-cartpole-v1"
    # alphas=['0.0', '0.5', '0.75', '1', '2', '5', '7.5', '10']
    env_name="continuous-mountain-car-v2"
    alphas=['0.0', '0.0025', '0.005', '0.0075', '0.009', '0.01', '0.02', '0.05']

    modes=['complete', 'state_var', 'mean_prop']
    seeds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    run_base_path = f"{DISPROD_RESULTS_PATH}/{env_name}/{run_name}/raw"
    env = env_name_mapping[env_name]

    statistics = {}

    results_base_path = f"{run_base_path}"

    for mode in modes:
        statistics[mode] = {}
        for alpha in alphas:
            statistics[mode][alpha] = {}
            path = glob.glob(f"{results_base_path}/06-14-2022-ablation-2-sogbofa_{mode}_{alpha}/results_*.txt")[0]
            with open(path, 'r') as f:
                data = f.readlines()
            mean, sd = [float(el.split(":")[1].strip()) for el in data[-1].strip("\n").split(",")]
            
            statistics[mode][alpha] = {"mean": mean, "sd": sd}


    with open(f"{results_base_path}/summary.txt", "w") as f:
        f.write(json.dumps(statistics))


    x = np.array(alphas)
    for mode in modes:
        mean = []
        sd = []
        for alpha in alphas:
            mean.append(statistics[mode][alpha]["mean"])
            sd.append(statistics[mode][alpha]["sd"])
        mean = np.array(mean)
        sd = np.array(sd)
        plt.plot(x, mean, label=mode_mapping[mode])
        plt.fill_between(x, mean-sd, mean+sd, alpha=0.2)
    plt.legend()
    plt.grid("on")
    plt.autoscale(tight=True)
    plt.xlabel(f"Alpha")
    plt.ylabel("Score")
    plt.title(f"Env: {env}")
    plt.tight_layout()
    plt.savefig(f"{run_base_path}/ablation.pdf", format='pdf', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()