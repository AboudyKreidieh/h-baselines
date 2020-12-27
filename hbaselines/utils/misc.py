"""Miscellaneous utility methods for this repository."""
import os
import errno
import functools
import inspect
import warnings


def ensure_dir(path):
    """Ensure that the directory specified exists, and if not, create it."""
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise  # pragma: no cover
    return path


def deprecated(base, new_path):
    """Print a deprecation warning.

    This is a decorator which can be used to mark functions as deprecated. It
    will result in a warning being emitted when the function is used.
    """
    def decorator(func1):
        if inspect.isclass(func1):
            fmt1 = "The class {base}.{name} is deprecated, use " \
                   "{new_path} instead."
        else:
            fmt1 = "The function {base}.{name} is deprecated, use " \
                   "{new_path} instead."

        @functools.wraps(func1)
        def new_func1(*args, **kwargs):
            warnings.simplefilter('always', PendingDeprecationWarning)
            warnings.warn(
                fmt1.format(
                    base=base,
                    name=func1.__name__,
                    new_path=new_path
                ),
                category=PendingDeprecationWarning,
                stacklevel=2
            )
            warnings.simplefilter('default', PendingDeprecationWarning)
            return func1(*args, **kwargs)

        return new_func1

    return decorator


def recursive_update(d, u):
    """Update a nested dictionary recursively recursively."""
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d
