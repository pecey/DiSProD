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


master_mapping = {"cartpole": {"env_name": "continuous-cartpole-v1",
                                "title": "Continuous Cart Pole",
                                "depths": [2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80]
                                },
                "mountain_car": {"env_name": "continuous-mountain-car-v2",
                                "title": "Continuous Mountain Car",
                                "depths": [20, 50, 75, 80, 90, 100, 110, 120, 130, 140, 150]
                                },
                "pendulum":     {"env_name": "pendulum-v2",
                                "title": "Pendulum",
                                "depths":[2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80]
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
    run_name = "08-17-2022-exp-depth"
    env = "mountain_car"

    noisy = True

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
    plot_file_name =  f"exp_depth_{env}.pdf" if not noisy else f"exp_depth_noisy_{env}.pdf"
    depths = master_mapping[env]["depths"]

    algorithms =["cem", "mppi", "disprod"]

    run_base_path = f"{DISPROD_RESULTS_PATH}/{env_name}/planning/{run_name}"

    statistics = {}

    results_base_path = f"{run_base_path}"

    print(f"Graph location: {results_base_path}/{plot_file_name}")

    for algorithm in algorithms:
        statistics[algorithm] = {}
        for depth in depths:
            statistics[algorithm][depth] = {}
            path = f"{results_base_path}/{run_name}-{algorithm}-{depth}/logs/summary.log"
            with open(path, 'r') as f:
                data = f.readlines()
            mean, sd = [float(el.split(":")[1].strip()) for el in data[-1].strip("\n").split(",")]
            statistics[algorithm][depth] = {"mean": mean, "sd": sd}


    with open(f"{run_base_path}/graph_summary.txt", "w") as f:
        f.write(json.dumps(statistics))


    x = np.array(depths)
    for algorithm in algorithms:
        mean = []
        sd = []
        for depth in depths:
            mean.append(statistics[algorithm][depth]["mean"])
            sd.append(statistics[algorithm][depth]["sd"])
        mean = np.array(mean)
        sd = np.array(sd)
        plt.plot(x, mean, label=alg_mapping[algorithm])
        plt.fill_between(x, mean-sd, mean+sd, alpha=0.2)
    plt.legend()
    plt.grid("on")
    plt.autoscale(tight=True)
    plt.xlabel(f"Planning horizon")
    plt.ylabel("Score")
    plt.title(f"Env: {title}")
    plt.tight_layout()
    plt.savefig(f"{run_base_path}/{plot_file_name}", format='pdf', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
