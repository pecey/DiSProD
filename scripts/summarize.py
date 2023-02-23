import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import json

DISPROD_PATH = os.getenv("DISPROD_PATH")
DISPROD_RESULTS_PATH = os.path.join(DISPROD_PATH, "results")

def main(args):
    path=args.path
    with open(f"{path}/summary.txt", "r") as f:
        data=f.readlines()

    scores = [float(el.split(",")[1].split(" ")[2]) for el in data]
    
    with open(f"{path}/summary_stats.txt", "w") as f:
        f.write(f"Mean: {np.mean(scores)}, SD: {np.std(scores)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()
    main(args)