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
    "pendulum-v2": "Pendulum"
}

alg_mapping = {
    "cem": "CEM",
    "mppi": "MPPI",
    "sogbofa": "DSSPD"
}

DISPROD_PATH = os.getenv("DISPROD_PATH")
DISPROD_RESULTS_PATH = os.path.join(DISPROD_PATH, "results")


def main(args):
    run_name = args.run_name
    env_name = args.env_name
    run_id = args.run_id
    x = args.x
    
    run_base_path = f"{DISPROD_RESULTS_PATH}/{env_name}/{run_name}/{run_id}"
    env = env_name_mapping[env_name]

    algorithms = [d for d in os.listdir(f"{run_base_path}") if os.path.isdir(f"{run_base_path}/{d}")]

    statistics = {"x": x}

    for algorithm in algorithms:
        mean, sd = [], []
        results_base_path = f"{run_base_path}/{algorithm}"
        for d in x:
            with open(f"{results_base_path}/{run_name}_{algorithm}_{d}.txt", 'r') as f:
                data = f.readlines()
                scores = data[-1]
                stats = [float(el.split(":")[1].strip()) for el in scores.split(",")]
                mean.append(stats[0])
                sd.append(stats[1])
        statistics[algorithm] = {"mean": mean, "sd": sd}


    with open(f"{run_base_path}/summary.txt", "w") as f:
        f.write(json.dumps(statistics))

    x = np.array(x)
    for algorithm in algorithms:
        mean = np.array(statistics[algorithm]["mean"])
        sd = np.array(statistics[algorithm]["sd"])
        plt.plot(x, mean, label=alg_mapping[algorithm])
        plt.fill_between(x, mean-sd, mean+sd, alpha=0.6)
    plt.legend()
    plt.grid("on")
    plt.autoscale(tight=True)
    plt.xlabel(f"{args.xlabel}")
    plt.ylabel("Score")
    plt.title(f"Env: {env}")
    plt.tight_layout()
    plt.savefig(f"{run_base_path}/{args.filename}.pdf", format='pdf', bbox_inches='tight')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=str, required=True)
    parser.add_argument('--run_name', type=str, required=True)
    parser.add_argument('--env_name', type=str, required=True)
    parser.add_argument('--filename', type=str, required=True)
    parser.add_argument('--xlabel', type=str, required=True)
    parser.add_argument('-x','--x', nargs='+', help='Values of x params', required=True)
    args = parser.parse_args()
    main(args)