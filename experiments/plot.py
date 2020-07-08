"""Utility method for plotting training performance."""
from collections import defaultdict
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import seaborn as sns

sns.set()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--files", type=str, nargs="+")
    parser.add_argument("--names", type=str, nargs="+")
    parser.add_argument("--out", type=str, default="out.png")
    parser.add_argument("--y", type=str, default="success_rate")
    parser.add_argument("--x", type=str, default="total_step")
    args = parser.parse_args()

    all_results = defaultdict(list)
    for path, name in zip(args.files, args.names):
        results = defaultdict(list)

        for filepath in tf.io.gfile.glob(os.path.join(path, "*/eval*.csv")):
            data = pd.read_csv(filepath)
            results[os.path.basename(filepath)[:-4]].append(data)

        for key in results.keys():
            data = pd.concat(results[key])
            data["name"] = name
            all_results[key].append(data)

    for key in all_results.keys():

        df = pd.concat(all_results[key])
        stop = df.groupby("name")["total_step"].max().min()
        df = df[df['total_step'] < stop]
        # df[args.x] = (df[args.x] / 10000).astype(int) * 10000
        plt.clf()
        ax = sns.lineplot(x=args.x, y=args.y, hue="name",
                          data=df)
        plt.savefig(args.out.replace(".", "_{}.".format(key)))
