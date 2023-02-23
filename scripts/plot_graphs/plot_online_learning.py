import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
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


master_mapping = {  "cartpole": {"env_name": "continuous-cartpole-v1", "title": "Continuous Cartpole"},
                    "pendulum": {"env_name": "pendulum-v2", "title": "Pendulum"}, 
                    "mountain_car": {"env_name": "continuous-mountain-car-v2", "title": "Continuous Mountain Car"},
                    "reacher": {"env_name": "reacher-3d-v1", "title": "Reacher"},
                    "pusher": {"env_name": "pusher-v2", "title": "Pusher"},
                    "half_cheetah": {"env_name": "half-cheetah-v1", "title": "Half Cheetah"},
                    
                }

alg_mapping = {
    "cem": "CEM",
    "mppi": "MPPI",
    "sogbofa": "DSSPD",
    "sogbofa-no-var": "DSSPD-NV"
}

DISPROD_PATH = os.getenv("DISPROD_PATH")
DISPROD_RESULTS_PATH = os.path.join(DISPROD_PATH, "results")

def plot_agg_curve(scores_dict, title, path):
    figure = plt.figure()

    for key in scores_dict.keys():
        scores = scores_dict[key]["mean"]
        scores_sd = scores_dict[key]["sd"]
        iterations = list(range(len(scores)))
        scores = np.array(scores)        
        plt.plot(iterations, scores, label=alg_mapping[key])
        plt.fill_between(iterations, scores - scores_sd, scores + scores_sd, alpha=0.2)

    plt.grid("on")
    plt.legend()
    plt.autoscale(tight=True)
    plt.xlabel("Iterations")
    plt.ylabel("Score")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path,  format='pdf', bbox_inches='tight')
    plt.close()

def plot_performance_curve(scores_dict, title, path):
    figure = plt.figure()

    for key in scores_dict.keys():
        scores = scores_dict[key]
        iterations = list(range(len(scores)))
        scores = np.array(scores)        
        plt.plot(iterations, scores, label=alg_mapping[key])

    plt.grid("on")
    plt.legend()
    plt.autoscale(tight=True)
    plt.xlabel("Iterations")
    plt.ylabel("Score")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path,  format='pdf', bbox_inches='tight')
    plt.close()

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def main():
    #run_name = "10-26-2022-noisy-online-learning"
    env = "reacher"
    env_name = master_mapping[env]["env_name"]
    env_title = master_mapping[env]["title"]

    seeds = [10, 20, 30, 40, 50]
    run_base_path = f"{DISPROD_RESULTS_PATH}/{env_name}/learning"
    graph_base_path = f"{DISPROD_RESULTS_PATH}/{env_name}/learning/10-26-2022-noisy-online-learning"

    data_paths = {"cem": f"{run_base_path}/10-15-2022-noisy-online-learning/10-15-2022-noisy-online-learning-cem", 
                  "sogbofa": f"{run_base_path}/10-18-2022-noisy-online-learning/10-18-2022-noisy-online-learning-sogbofa",
                  "sogbofa-no-var": f"{run_base_path}/10-26-2022-noisy-online-learning/10-26-2022-noisy-online-learning-sogbofa"}
    
    
    # algorithms =["cem", "mppi", "sogbofa"]
    algorithms=["cem", "sogbofa", "sogbofa-no-var"]
    statistics_actual = {seed: {} for seed in seeds}
    statistics_smooth = {seed: {} for seed in seeds}
    statistics_moving_avg = {seed: {} for seed in seeds}

    for seed in seeds:
        
        title = f"Noisy {env_title}(Alpha = 1) - Seed: {seed}"

        path_actual = f"{graph_base_path}/performance_curve_actual_{seed}.pdf"
        path_smooth = f"{graph_base_path}/performance_curve_smooth_{seed}.pdf"
        path_moving_avg =  f"{graph_base_path}/performance_curve_ma_{seed}.pdf"
        
        for algorithm in algorithms:
            data_path = f"{data_paths[algorithm]}_{seed}/logs.mat"
            if os.path.exists(data_path):
                data = loadmat(data_path)
                statistics_actual[seed][algorithm] = data["agg_rewards"][0][:45]
                statistics_smooth[seed][algorithm] = np.maximum.accumulate(data["agg_rewards"][0][:45], axis=-1)
                statistics_moving_avg[seed][algorithm] = np.concatenate((data["agg_rewards"][0][:4], moving_average(data["agg_rewards"][0][:45], 5)))

        plot_performance_curve(statistics_actual[seed], f"{title} - True Returns", path_actual)
        plot_performance_curve(statistics_smooth[seed], f"{title} - Smoothened Returns", path_smooth)
        plot_performance_curve(statistics_moving_avg[seed], f"{title} - Moving Average", path_moving_avg)

    aggregate_actual = {}
    aggregate_moving_avg = {}

    for algorithm in statistics_actual[seeds[0]].keys():
        actual_scores = np.vstack([statistics_actual[seed][algorithm] for seed in seeds])
        aggregate_actual[algorithm] = {"mean": np.mean(actual_scores, axis=0), "sd": np.std(actual_scores, axis=0)}

        moving_avg_scores = np.vstack([statistics_moving_avg[seed][algorithm] for seed in seeds])
        aggregate_moving_avg[algorithm] = {"mean": np.mean(moving_avg_scores, axis=0), "sd": np.std(moving_avg_scores, axis=0)}

    plot_agg_curve(aggregate_actual, f"Noisy {env_title} (Alpha = 1) - True Returns - Aggregate", f"{graph_base_path}/performance_curve_actual_aggregate.pdf")
    plot_agg_curve(aggregate_moving_avg, f"Noisy {env_title} (Alpha = 1) - Moving Average - Aggregate", f"{graph_base_path}/performance_curve_ma_aggregate.pdf")

if __name__ == "__main__":
    main()
