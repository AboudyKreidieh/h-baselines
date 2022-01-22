#!/usr/bin/env python3
# flake8: noqa
"""Setup script for the h-baselines repository."""
import os
import subprocess
from setuptools.command.install import install
from os.path import dirname
from os.path import realpath
from setuptools import find_packages
from setuptools import setup
from setuptools import Distribution

from hbaselines.version import __version__


def _read_requirements_file():
    req_file_path = '%s/requirements.txt' % dirname(realpath(__file__))
    with open(req_file_path) as f:
        return [line.strip() for line in f]


class CustomInstallCommand(install):
    """Customized setuptools install command."""

    def run(self):
        """See parent class."""
        # Install ray[tune].
        subprocess.check_call(['pip', 'install', 'ray[tune]'])

        # Unzip files.
        subprocess.check_call([
            'unzip',
            os.path.join(
                dirname(realpath(__file__)),
                'experiments/warmup/highway.zip',
            )
        ])
        subprocess.check_call([
            'unzip',
            os.path.join(
                dirname(realpath(__file__)),
                'experiments/warmup/i210.zip',
            )
        ])

        install.run(self)


class BinaryDistribution(Distribution):
    """See parent class."""

    @staticmethod
    def has_ext_modules():
        """Return True for external modules."""
        return True


setup(
    name='h-baselines',
    version=__version__,
    distclass=BinaryDistribution,
    cmdclass={"install": CustomInstallCommand},
    packages=find_packages(),
    install_requires=_read_requirements_file(),
    description='h-baselines: a repository of high-performing and benchmarked '
                'hierarchical reinforcement learning models and algorithm',
    author='Aboudy Kreidieh',
    url='https://github.com/AboudyKreidieh/h-baselines',
    author_email='aboudy@berkeley.edu',
    zip_safe=False,
)
