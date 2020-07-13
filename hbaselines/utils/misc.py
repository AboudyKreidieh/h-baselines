"""Miscellaneous utility methods for this repository."""
import os
import errno
import numpy as np


def ensure_dir(path):
    """Ensure that the directory specified exists, and if not, create it."""
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise  # pragma: no cover
    return path


def explained_variance(y_pred, y_true):
    """Compute fraction of variance that ypred explains about y.

    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    Parameters
    ----------
    y_pred : np.ndarray
        the prediction
    y_true : np.ndarray
        the expected value

    Returns
    -------
    float
        explained variance of ypred and y
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
