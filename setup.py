#!/usr/bin/env python3
# flake8: noqa
from os.path import dirname, realpath
from setuptools import find_packages, setup, Distribution
import setuptools.command.build_ext as _build_ext
import subprocess
from hbaselines.version import __version__


def _read_requirements_file():
    req_file_path = '%s/requirements.txt' % dirname(realpath(__file__))
    with open(req_file_path) as f:
        return [line.strip() for line in f]


class build_ext(_build_ext.build_ext):
    def run(self):
        try:
            import traci
        except ImportError:
            subprocess.check_call(
                ['pip', 'install',
                 'https://akreidieh.s3.amazonaws.com/sumo/flow-0.2.0/'
                 'sumotools-0.1.0-py3-none-any.whl'])


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(
    name='hbaselines',
    version=__version__,
    packages=find_packages(),
    install_requires=_read_requirements_file(),
    zip_safe=False,
)
