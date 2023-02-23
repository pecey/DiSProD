import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.image as img
import numpy as np
import argparse
import os
import json
import glob
import shutil

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


DISPROD_PATH = os.path.join(os.getenv("HOME"), "awesome-sogbofa")
DISPROD_RESULTS_PATH = os.path.join(DISPROD_PATH, "results")

def plot_trajectory_from_json(json_paths, alias, plot_path):
    n = len(json_paths)
    f, ax = plt.subplots(1, n, figsize=(12*n,8))
    for idx, json_path in enumerate(json_paths):
        with open(json_path, "r") as f:
            data = json.loads(f.readlines()[0])

            ax[idx].set_xlim(data['x_lim'][0], data['x_lim'][1])
            ax[idx].set_ylim(data['y_lim'][0], data['y_lim'][1])

            ax[idx].plot(data['tau'][0], data['tau'][1], '--', alpha=0.8)
            ax[idx].plot(data['position'][0], data['position'][1], 'o')

            ax[idx].plot(data['goal'][0], data['goal'][1], "gx")
            boundary = plt.Circle((data['goal'][0], data['goal'][1]), radius=data['goal_boundary'][0], color='orange', alpha=0.8)
            ax[idx].add_artist(boundary)

            if "obstacles" in data.keys():
                for obstacle in data["obstacles"]:
                    ax[idx].add_patch(Rectangle((obstacle["x"], obstacle["y"]), obstacle["width"], obstacle["height"], color="brown"))

            if "boundary" in data.keys():
                for boundary in data["boundary"]:
                    ax[idx].add_patch(
                        Rectangle((boundary["x"], boundary["y"]), boundary["width"], boundary["height"], color='black'))

            ax[idx].set_title(alias[idx])

                
    plt.grid()
    plt.savefig(plot_path)
    plt.close()


def main():
    run_id = "1656633492"
    result_path = f"/home/pecey/Research/awesome-sogbofa/results/continuous-dubins-car-w-velocity-state-v0/planning/06-30-2022-sogbofa-ablation-{run_id}"
    os.makedirs(result_path, exist_ok=True)

    paths = [f"/home/pecey/Research/awesome-sogbofa/results/continuous-dubins-car-w-velocity-state-v0/planning/06-30-2022-sogbofa-complete-suite-0.5-no_var-{run_id}",
            f"/home/pecey/Research/awesome-sogbofa/results/continuous-dubins-car-w-velocity-state-v0/planning/06-30-2022-sogbofa-complete-suite-0.5-state_var_only-{run_id}",
            f"/home/pecey/Research/awesome-sogbofa/results/continuous-dubins-car-w-velocity-state-v0/planning/06-30-2022-sogbofa-complete-suite-0.5-action_var_only-{run_id}",
            f"/home/pecey/Research/awesome-sogbofa/results/continuous-dubins-car-w-velocity-state-v0/planning/06-30-2022-sogbofa-complete-suite-0.5-complete-{run_id}"]

    # maps = ["no-ob-1", "no-ob-2", "no-ob-3", "no-ob-4", "no-ob-5", "ob-1", "ob-2", "ob-3", "ob-4", "ob-5", "ob-6", "ob-7", "ob-8", "ob-9", "ob-10", "ob-11", "u", "cave-mini"]
    maps = ["cave-mini"]

    alias = ["No variance", "Using only state variance", "Using only action variance", "Complete expression"]

    seeds = [0, 42, 84, 126, 168, 210, 252, 294, 336, 378]

    for seed in seeds:
        for map in maps:
            json_paths = [f"{path}/dubins_suite_{map}/graphs/{seed}_trajectory.json" for path in paths]
            plot_trajectory_from_json(json_paths, alias, f"{result_path}/{map}_{seed}.pdf")

    for path in paths:
        shutil.move(path, result_path)

if __name__ == "__main__":
    main()