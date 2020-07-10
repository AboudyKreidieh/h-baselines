"""Utility method for plotting training performance."""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import sys
from copy import deepcopy

import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 1.5  # set the value globally

COLORS = [
    (0, 0, 255),
    (255, 129, 19),
    (157, 116, 195),
    (52, 164, 52),
    (144, 93, 82),
    (255, 0, 0),
]
for indx, c in enumerate(COLORS):
    r, g, b = c
    COLORS[indx] = r / 256, g / 256, b / 256


def parse_options(args):
    """Parse plotting options user can specify in command line."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "folders", type=str, nargs="+",
        help="the path to the folders containing results from a similar "
             "choice of model/algorithm. The mean and standard deviation of "
             "all these result will be plotted on the graph.")

    # optional arguments
    parser.add_argument(
        "--names", type=str, nargs="+", default=None,
        help="The names to be assigned for each result. Must be equal in "
             "size to the number of folder specified. If not assigned, no "
             "legend is added.")
    parser.add_argument(
        "--out", type=str, default="out.png",
        help="the path where the figure should be saved. Append with a .svg "
             "file if you would like to generate SVG formatted graphs.")
    parser.add_argument(
        "--use_eval", action="store_true",
        help="whether to use the eval_*.csv or train.csv files to generate "
             "the plots")
    parser.add_argument(
        "--y", type=str, default="rollout/return_history",
        help="the column to use for the y-coordinates")
    parser.add_argument(
        "--x", type=str, default="total/steps",
        help="the column to use for the x-coordinates")
    parser.add_argument(
        "--ylabel", type=str, default=None,
        help="the label to use for the y-axis. If set to None, the name of "
             "the column used for the y-coordinates is used.")
    parser.add_argument(
        "--xlabel", type=str, default=None,
        help="the label to use for the x-axis. If set to None, the name of "
             "the column used for the x-coordinates is used.")
    parser.add_argument(
        "--show", action="store_true",
        help="whether to show the figure that was saved")

    return parser.parse_args(args)


def import_results(folders, x, y, use_eval):
    """Import relevant data from each logging file in the specified folders.

    Parameters
    ----------
    folders : list of str
        the path to the folders containing results from a similar choice of
        model/algorithm. The mean and standard deviation of all these result
        will be plotted on the graph.
    x : str
        the column to use for the x-coordinates
    y : str
        the column to use for the y-coordinates
    use_eval : bool
        whether to use the eval_*.csv or train.csv files to generate the plots

    Returns
    -------
    np.ndarray
        the steps numbers. Used as the x-coordinate in the graph.
    list of np.ndarray
        a list of mean returns from each model/algorithm at every step
    list of np.ndarray
        a list of standard deviation of the returns from each model/algorithm
        at every step
    """
    res_x = np.array([])
    mean = []
    std = []

    for folder in folders:
        res_i = []

        # Collect the names of all the sub-directories. These directories
        # should contain the logging information.
        sub_directories = os.listdir(folder)

        # Append the data from each sub-directory to the per-folder results.
        if use_eval:
            # Determine the number of evaluation files.
            total_eval = len([
                x for x in os.listdir(os.path.join(folder, sub_directories[0]))
                if x.startswith("eval")])
            res_i = [[] for _ in range(total_eval)]
            res_mean_i = [0 for _ in range(total_eval)]
            res_std_i = [0 for _ in range(total_eval)]

            # Get the x and y data for each sub-directory.
            for dir_i in sub_directories:
                for eval_num in range(total_eval):
                    df = pd.read_csv(os.path.join(
                        folder,
                        os.path.join(dir_i, "eval_{}.csv".format(eval_num))
                    ))
                    res_x = choose_x(res_x, np.array(df[x]))
                    res_i[eval_num].append(np.array(df[y]))

            for eval_num in range(total_eval):
                # Shrink every array to be the length of the smallest element.
                min_length = res_x.shape[0]
                for i in range(len(res_i[eval_num])):
                    res_i[eval_num][i] = res_i[eval_num][i][:min_length]
                res_i[eval_num] = np.asarray(res_i[eval_num])

                # Compute the mean and std.
                res_mean_i[eval_num] = np.mean(res_i[eval_num], axis=0)
                res_std_i[eval_num] = np.std(res_i[eval_num], axis=0)
        else:
            # Get the x and y data for each sub-directory.
            for dir_i in sub_directories:
                df = pd.read_csv(os.path.join(
                    folder, os.path.join(dir_i, "train.csv")))
                res_x = choose_x(res_x, np.array(df[x]))
                res_i.append(np.array(df[y]))

            # Shrink every array to be the length of the smallest element.
            min_length = res_x.shape[0]
            for i in range(len(res_i)):
                res_i[i] = res_i[i][:min_length]
            res_i = np.asarray(res_i)

            # Compute the mean and std.
            res_mean_i = np.mean(res_i, axis=0)
            res_std_i = np.std(res_i, axis=0)

        # Add to the mean and std to eh list of all results.
        mean.append(res_mean_i)
        std.append(res_std_i)

    return res_x, mean, std


def choose_x(x, x_new):
    """Choice the x array with the smallest number of elements."""
    if x.shape[0] == 0:
        # for initial empty array
        return x_new
    else:
        return x if x.shape[0] < x_new.shape[0] else x_new


def plot_fig(mean,
             std,
             steps,
             xlabel,
             ylabel,
             y_lim=None,
             name="results",
             legend=None,
             show=False,
             save=False):
    """Plot the mean/std of the different models/algorithms.

    Parameters
    ----------
    mean : list of np.ndarray
        a list of mean returns from each model/algorithm at every step
    std : list of np.ndarray
        a list of standard deviation of the returns from each model/algorithm
        at every step
    steps : np.ndarray
        the steps numbers. Used as the x-coordinate
    xlabel : str
        the x-label in the figure
    ylabel : str
        the y-label in the figure
    y_lim : list of float or None
        the bounds for the y axis. Unspecified by default.
    name : str
        the path to save the file to
    legend : list of str or None
        the names of the elements in the legend
    show : bool
        whether to show the figure that was generated
    save : bool
        whether to save the figure
    """
    colors = deepcopy(COLORS)
    plt.figure(figsize=(8, 5))
    for mean_i, std_i, color in zip(mean, std, colors):
        plt.plot(steps, mean_i, c=color, lw=2)
        plt.fill_between(
            steps, mean_i + std_i, mean_i - std_i, facecolor=color, alpha=0.2)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.ylim(y_lim)
    plt.grid(linestyle='dotted', lw=1)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 4))
    if legend is not None:
        plt.legend(legend)
    if save:
        plt.savefig(name, bbox_inches='tight', transparent=True)
    if show:
        plt.show()


if __name__ == "__main__":
    flags = parse_options(sys.argv[1:])

    # Collect the relevant data.
    res_steps, res_mean, res_std = import_results(
        folders=flags.folders,
        x=flags.x,
        y=flags.y,
        use_eval=flags.use_eval,
    )

    # Plot the results.
    if isinstance(res_mean[0][0], float):
        plot_fig(
            mean=res_mean,
            std=res_std,
            steps=res_steps,
            y_lim=None,
            name=flags.out,
            legend=flags.names,
            show=flags.show,
            xlabel=flags.xlabel or flags.x,
            ylabel=flags.ylabel or flags.y,
            save=True,
        )
    else:
        for eval_i in range(len(res_mean[0])):
            plot_fig(
                mean=[x[eval_i] for x in res_mean],
                std=[x[eval_i] for x in res_std],
                steps=res_steps,
                y_lim=None,
                name="{}_{}".format(eval_i, flags.out),
                legend=flags.names,
                show=flags.show,
                xlabel=flags.xlabel or flags.x,
                ylabel=flags.ylabel or flags.y,
                save=True,
            )
