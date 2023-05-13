import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
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


master_mapping = {"mountain_car": {"env_name": "continuous-mountain-car-v3",
                                "title": "Continuous Mountain Car",
                                "n_actions": [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
                                }
                }

alg_mapping = {
    "cem": "CEM",
    "mppi": "MPPI",
    "disprod": "DiSProD"
}

color_mapping = {
    "disprod 200": "#2ca02c",
    "cem 200": "#1f77b4",
    "cem 2000": "#d62728",
    "cem 20000": "#9467bd"
}

DISPROD_PATH = os.getenv("DISPROD_PATH")
DISPROD_RESULTS_PATH = os.path.join(DISPROD_PATH, "results")


def main():
    env = "mountain_car"
    run_names = {
        "cem": "01-16-2023-exp-high-dim", 
        "disprod" : "12-03-2022-exp-high-dim"}
    algorithms = {
        "cem": [200, 2000, 20000], 
        "disprod": [200],
        } 

    run_base_path = f"{DISPROD_RESULTS_PATH}/{env_name}/planning"
    output_base_path = f"{run_base_path}/01-16-2023-exp-high-dim"

    # noisy = True
	
    samples = [200, 2000, 20000]
   

    env_name = master_mapping[env]["env_name"]
    title = master_mapping[env]["title"] #if not noisy else f'Noisy {master_mapping[env]["title"]} -  (Alpha = {alpha_val})'
    plot_file_name =  f"exp_high_dim_{env}_scores.pdf" #if not noisy else f"exp_samples_noisy_{env}.pdf"
    steps_file_name = f"exp_high_dim_{env}_steps.pdf"
    n_actions = master_mapping[env]["n_actions"]   

    statistics = {}

    for algorithm, samples in algorithms.items():
        run_name = run_names[algorithm]
        results_base_path = f"{run_base_path}/{run_name}"

        for sample in samples:
            statistics[f"{algorithm} {sample}"] = {}
            for n_action in n_actions:
                statistics[f"{algorithm} {sample}"][n_action] = {}
                summary_path = f"{results_base_path}/{run_name}-{algorithm}_{n_action}_{sample}/logs/summary.log"
                output_path = f"{results_base_path}/{run_name}-{algorithm}_{n_action}_{sample}/logs/output.log"
                if algorithm == "disprod":
                    summary_path = summary_path.replace("_200","")
                    output_path = output_path.replace("_200","")
                with open(summary_path, 'r') as f:
                    data = f.readlines()
                mean, sd = [float(el.split(":")[1].strip()) for el in data[-1].strip("\n").split(",")]

                with open(output_path, 'r') as f:
                    data = json.load(f)
                steps_mean = np.mean([e[0]['steps'] for e in data])
                steps_sd = np.std([e[0]['steps'] for e in data])

                statistics[f"{algorithm} {sample}"][n_action] = {"mean": mean, "sd": sd, 'steps_mean': steps_mean, 'steps_sd': steps_sd}

    print(f"Graph location: {results_base_path}/{plot_file_name}")


    with open(f"{output_base_path}/graph_summary.txt", "w") as f:
        f.write(json.dumps(statistics))


    figure = plt.figure()
    x = np.array(n_actions)
    for algorithm, samples in algorithms.items():
        for sample in samples:
            mean = []
            sd = []
            for n_action in n_actions:
                mean.append(statistics[f"{algorithm} {sample}"][n_action]["mean"])
                sd.append(statistics[f"{algorithm} {sample}"][n_action]["sd"])
            mean = np.array(mean)
            sd = np.array(sd)
            plt.plot(x, mean, label=f"{alg_mapping[algorithm]}-{sample}", color = color_mapping[f"{algorithm} {sample}"])
            plt.fill_between(x, mean-sd, mean+sd, alpha=0.2, color = color_mapping[f"{algorithm} {sample}"])
    plt.legend(loc="lower left")
    plt.grid("on")
    plt.autoscale(tight=True)
    plt.xlabel(f"Number of actions")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(f"{output_base_path}/{plot_file_name}", format='pdf', bbox_inches='tight')
    plt.close()
    
    figure = plt.figure()
    x = np.array(n_actions)
    for algorithm, samples in algorithms.items():
        for sample in samples:
            mean = []
            sd = []
            for n_action in n_actions:
                mean.append(statistics[f"{algorithm} {sample}"][n_action]["steps_mean"])
                sd.append(statistics[f"{algorithm} {sample}"][n_action]["steps_sd"])
            mean = np.array(mean)
            sd = np.array(sd)
            plt.plot(x, mean, label=f"{alg_mapping[algorithm]}-{sample}", color = color_mapping[f"{algorithm} {sample}"])
            plt.fill_between(x, mean-sd, mean+sd, alpha=0.2, color = color_mapping[f"{algorithm} {sample}"])
    #plt.legend()
    plt.grid("on")
    plt.autoscale(tight=True)
    plt.xlabel(f"Number of actions")
    plt.ylabel("Steps Taken")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{output_base_path}/{steps_file_name}", format='pdf', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
