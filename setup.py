#!/usr/bin/env python3
# flake8: noqa
"""Setup script for the h-baselines repository."""
from os.path import dirname, realpath
from setuptools import find_packages, setup
from hbaselines.version import __version__


def _read_requirements_file():
    req_file_path = '%s/requirements.txt' % dirname(realpath(__file__))
    with open(req_file_path) as f:
        return [line.strip() for line in f]


setup(
    name='hbaselines',
    version=__version__,
    packages=find_packages(),
    install_requires=_read_requirements_file(),
    zip_safe=False,
)
