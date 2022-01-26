#!/usr/bin/env python3
# flake8: noqa
"""Setup script for the h-baselines repository."""
import os
from zipfile import ZipFile
from setuptools.command.install import install
from os.path import dirname
from os.path import realpath
from setuptools import setup

from hbaselines.version import __version__


def _read_requirements_file():
    req_file_path = '%s/requirements.txt' % dirname(realpath(__file__))
    with open(req_file_path) as f:
        return [line.strip() for line in f]


class CustomInstall(install):
    """Custom installation procedure."""

    def __init__(self, dist):
        super(install, self).__init__(dist)
        self.__post_install()

    def run(self):
        """See parent class."""
        install.run(self)

    @staticmethod
    def __post_install():
        directory = os.path.join(
            dirname(realpath(__file__)), 'experiments/warmup')

        # Unzip files.
        with ZipFile(os.path.join(directory, 'highway.zip'), 'r') as f:
            f.extractall(directory)
        with ZipFile(os.path.join(directory, 'i210.zip'), 'r') as f:
            f.extractall(directory)


setup(
    name='h-baselines',
    version=__version__,
    cmdclass={"install": CustomInstall},
    install_requires=_read_requirements_file(),
    description='h-baselines: a repository of high-performing and benchmarked '
                'hierarchical reinforcement learning models and algorithm',
    author='Aboudy Kreidieh',
    url='https://github.com/AboudyKreidieh/h-baselines',
    author_email='aboudy@berkeley.edu',
)
