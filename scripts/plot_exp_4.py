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
    run_name = "06-14-2022-reward-sparsity-vs-performance-2000-restarts"
    env_name = "continuous-mountain-car-v2"
    folder_name = "06-14-2022-exp4-2000-restarts"

    sparsity_values=['0.1', '1', '2', '3', '4', '5', '7.5', '10']
    depths = [100, 150, 200]

    d_name_mapping = {100: "100", 150:  "150", 200: "200"}

    run_base_path = f"{DISPROD_RESULTS_PATH}/{env_name}/{run_name}"
    env = env_name_mapping[env_name]

    algorithms = ["cem", "mppi", "sogbofa"]
    
    results_base_path = f"{run_base_path}"

    for depth in depths:
        statistics = {}
        for algorithm in algorithms:
            statistics[algorithm] = {}
            for sparsity_value in sparsity_values:
                statistics[algorithm][sparsity_value] = {}
                filename = glob.glob(f"{results_base_path}/{depth}/{folder_name}-{algorithm}_{d_name_mapping[depth]}_{sparsity_value}/results*.txt")[0]
                with open(filename, 'r') as f:
                    data = f.readlines()
                mean, sd = [float(el.split(":")[1].strip()) for el in data[-1].strip("\n").split(",")]
                statistics[algorithm][sparsity_value] = {"mean": mean, "sd": sd}


        with open(f"{run_base_path}/summary_{depth}.txt", "w") as f:
            f.write(json.dumps(statistics))


        x = np.array(sparsity_values)
        for algorithm in algorithms:
            mean = []
            sd = []
            for sparsity_value in sparsity_values:
                mean.append(statistics[algorithm][sparsity_value]["mean"])
                sd.append(statistics[algorithm][sparsity_value]["sd"])
            mean = np.array(mean)
            sd = np.array(sd)
            plt.plot(x, mean, label=alg_mapping[algorithm])
            plt.fill_between(x, mean-sd, mean+sd, alpha=0.2)
        plt.legend()
        plt.grid("on")
        plt.autoscale(tight=True)
        plt.xlabel(f"Sparsity Multiplier")
        plt.ylabel("Score")
        plt.title(f"Env: {env} - Horizon: {depth}")
        plt.tight_layout()
        plt.savefig(f"{run_base_path}/exp4_{depth}.pdf", format='pdf', bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    main()