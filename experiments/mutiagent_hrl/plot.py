import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_results_labels(dir_path):
    """Retrieve the results and names for each training curve.

    blank

    Parameters
    ----------
    dir_path : str
        location of the results directories

    Returns
    -------
    list of np.ndarray
        the average cumulative return for each training curve
    list of str
        name of each training curve
    int
        change in number of steps in between results values
    """
    dx = 0
    directories = [dI for dI in os.listdir(dir_path)
                   if os.path.isdir(os.path.join(dir_path, dI))]
    print(directories)
    results = []

    for directory in directories:
        res = None
        for i in range(3):
            path = os.path.join(dir_path, directory,
                                'results_{}.csv'.format(i))
            data = pd.read_csv(path)
            if i == 0:
                res = np.zeros((len(data['total/steps']), 3))
                dx = data['total/steps'][2] - data['total/steps'][1]
            res[:, i] = data['rollout/return_history']
        results.append(res.T)

    return results, directories, dx


def plot_results(dir_path):
    """Plot the training curves of multiple algorithms.

    Parameters
    ----------
    dir_path : str
        location of the results directories
    """
    results, labels, dx = get_results_labels(dir_path)

    colors = plt.cm.get_cmap('tab10', len(labels)+1)
    fig = plt.figure(figsize=(16, 9))
    for i, (label, result) in enumerate(zip(labels, results)):
        plt.plot(np.arange(result.shape[1]) * dx, np.mean(result, 0),
                 color=colors(i), linewidth=2, label=label)
        plt.fill_between(np.arange(len(result[0])) * dx,
                         np.mean(result, 0) - np.std(result, 0),
                         np.mean(result, 0) + np.std(result, 0),
                         alpha=0.25, color=colors(i))
    plt.title('Training Performance of Different Algorithms', fontsize=25)
    plt.ylabel('Cumulative return', fontsize=20)
    plt.xlabel('Number of steps', fontsize=20)
    plt.xlim([0, results[0].shape[1] * dx])
    plt.tick_params(labelsize=15)
    plt.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))
    plt.legend(fontsize=20)
    plt.grid(linestyle=':')
    plt.show()

    return fig


def create_parser():
    """Generate the parser for the plotting operation."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Creates plots for the reward curves within a folder.',
        epilog='python plot.py /path/to/results_dir')

    # required input parameters
    parser.add_argument('dir_path', type=str,
                        help='Directory with the results folder(s).')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    plot_results(args.dir_path)
