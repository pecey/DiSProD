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


master_mapping = {"mountain_car": {"env_name": "sparse-continuous-mountain-car-v1",
                                "title": "Continuous Mountain Car",
                                "sparsity_values": ['0.1','0.2','0.5','1','2','3','4']
                                }
                }

alg_mapping = {
    "cem": "CEM",
    "mppi": "MPPI",
    "sogbofa": "DSSPD"
}

DISPROD_PATH = os.getenv("DISPROD_PATH")
DISPROD_RESULTS_PATH = os.path.join(DISPROD_PATH, "results")


def main():
    run_name = "08-22-2022-exp-reward-sparsity"
    env = "mountain_car"
    depth = 200

    env_name = master_mapping[env]["env_name"]
    title = master_mapping[env]["title"]
    sparsity_values = master_mapping[env]["sparsity_values"]

    algorithms =["cem", "mppi", "sogbofa"]

    run_base_path = f"/N/u/palchatt/BigRed3/awesome-sogbofa/results/{env_name}/planning/{run_name}"

    statistics = {}

    results_base_path = f"{run_base_path}"

    for algorithm in algorithms:
        statistics[algorithm] = {}
        for value in sparsity_values:
            statistics[algorithm][value] = {}
            path = f"{results_base_path}/{run_name}-{algorithm}-{depth}-{value}/logs/summary.log"
            with open(path, 'r') as f:
                data = f.readlines()
            mean, sd = [float(el.split(":")[1].strip()) for el in data[-1].strip("\n").split(",")]
            statistics[algorithm][value] = {"mean": mean, "sd": sd}


    with open(f"{run_base_path}/graph_summary.txt", "w") as f:
        f.write(json.dumps(statistics))


    x = np.array(sparsity_values)
    for algorithm in algorithms:
        mean = []
        sd = []
        for value in sparsity_values:
            mean.append(statistics[algorithm][value]["mean"])
            sd.append(statistics[algorithm][value]["sd"])
        mean = np.array(mean)
        sd = np.array(sd)
        plt.plot(x, mean, label=alg_mapping[algorithm])
        plt.fill_between(x, mean-sd, mean+sd, alpha=0.2)
    plt.legend()
    plt.grid("on")
    plt.autoscale(tight=True)
    plt.xlabel(f"Sparsity Multiplier")
    plt.ylabel("Score")
    plt.title(f"Env: {title} - Horizon: {depth}")
    plt.tight_layout()
    plt.savefig(f"{run_base_path}/exp_reward_sparsity_{env}_{depth}.pdf", format='pdf', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
