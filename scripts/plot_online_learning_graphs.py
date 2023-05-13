import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import json

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
    "pendulum-v2": "Pendulum",
    "pusher-v2": "Pusher"
}

alg_mapping = {
    "cem": "CEM",
    "mppi": "MPPI",
     "disprod": "DiSProD"
}

DISPROD_PATH = os.getenv("DISPROD_PATH")
DISPROD_RESULTS_PATH = os.path.join(DISPROD_PATH, "results")


def main():
    run_name = "06-11-2022-online-learning"
    env_name = "continuous-cartpole-v1"

    x = list(range(1,26))
    seeds = [10, 20, 30, 40, 50]
    run_base_path = f"{DISPROD_RESULTS_PATH}/{env_name}/learning/{run_name}"
    env = env_name_mapping[env_name]

    algorithms = os.listdir(f"{run_base_path}")

    statistics = {"x": x}

    for seed in seeds:
        for algorithm in algorithms:
            mean, sd = [], []
            results_base_path = f"{run_base_path}/{algorithm}"
            with open(f"{results_base_path}/seed_{seed}.txt", 'r') as f:
                data = f.readlines()
                for line_idx in range(len(x)):
                    info_ = data[line_idx]
                    scores = info_.split(":")[2].strip()
                    stats = [float(el.strip().split(" ")[1]) for el in scores.split(",")]
                    mean.append(stats[0])
                    sd.append(stats[1])
            statistics[algorithm] = {"mean": mean, "sd": sd}


        with open(f"{run_base_path}/summary_{seed}.txt", "w") as f:
            f.write(json.dumps(statistics))


        x = np.array(x)
        for algorithm in algorithms:
            mean = np.array(statistics[algorithm]["mean"])
            sd = np.array(statistics[algorithm]["sd"])
            plt.plot(x, mean, label=alg_mapping[algorithm])
            plt.fill_between(x, mean-sd, mean+sd, alpha=0.2)
        plt.legend()
        plt.grid("on")
        plt.autoscale(tight=True)
        plt.xlabel(f"Iterations")
        plt.ylabel("Score")
        plt.title(f"Env: {env}")
        plt.tight_layout()
        plt.savefig(f"{run_base_path}/online_learning_{seed}.pdf", format='pdf', bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--run_id', type=str, required=True)
    # parser.add_argument('--run_name', type=str, required=True)
    # parser.add_argument('--env_name', type=str, required=True)
    # parser.add_argument('--filename', type=str, required=True)
    # parser.add_argument('--xlabel', type=str, required=True)
    # parser.add_argument('-x','--x', nargs='+', help='Values of x params', required=True)
    # args = parser.parse_args()
    # main(args)
    main()