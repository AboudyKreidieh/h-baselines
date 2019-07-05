[![Build Status](https://travis-ci.com/AboudyKreidieh/h-baselines.svg?branch=master)](https://travis-ci.com/AboudyKreidieh/h-baselines)
[![Coverage Status](https://coveralls.io/repos/github/AboudyKreidieh/h-baselines/badge.svg?branch=master)](https://coveralls.io/github/AboudyKreidieh/h-baselines?branch=master)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/AboudyKreidieh/h-baselines/blob/master/LICENSE)

# h-baselines

blank

## Contents

* [Installation](#installation)
* [Supported Models/Algorithms](#supported-modelsalgorithms)
  * [FuN (FeUdal Networks for Hierarchical Reinforcement Learning)](#fun-feudal-networks-for-hierarchical-reinforcement-learning)
  * [HIRO (Data Efficient Hierarchical Reinforcement Learning)](#hiro-data-efficient-hierarchical-reinforcement-learning)
  * [HAC (Learning Multi-Level Hierarchies with Hindsight)](#hac-learning-multi-level-hierarchies-with-hindsight)
* [Environments](#environments)
  * [MuJoCo Environments](#mujoco-environments)
  * [Mixed Autonomy Traffic](#mixed-autonomy-traffic)
* [Citing](#citing)
* [Bibliography](#bibliography)
* [Useful Links](#useful-links)

## Installation

To install the h-baselines repository, begin by opening a terminal and set the
working directory of the terminal to match

```bash
cd path/to/h-baselines
```

Next, create and activate a conda environment for this repository by running 
the commands in the script below. Note that this is not required, but highly 
recommended. If you do not have Anaconda on your device, refer to the provided
links to install either [Anaconda](https://www.anaconda.com/download) or
[Miniconda](https://conda.io/miniconda.html).

```bash
conda env create -f environment.yml
source activate h-baselines
```

Finally, install the contents of the repository onto your conda environment (or
your local python build) by running the following command:

```bash
pip install -e .
```

If you would like to (optionally) validate that the repository was successfully
installed and is running, you can do so by executing the unit tests as follows:

```bash
nose2
```

The test should return a message along the lines of:

    ----------------------------------------------------------------------
    Ran XXX tests in YYYs

    OK

## Supported Models/Algorithms

blank

### FuN (FeUdal Networks for Hierarchical Reinforcement Learning)

One of the early works on feudal variants of hierarchical reinforcement 
learning since the surge of deep neural networks as a viable tool in machine
learning, this model attempts to adapt more modern machine learning techniques
to the original model presented by [1].

### HIRO (Data Efficient Hierarchical Reinforcement Learning)

blank

### HAC (Learning Multi-Level Hierarchies with Hindsight)

blank

## Environments

This repository contains multiple 

### MuJoCo Environments

blank

**Pendulum** blank

**Ant Push** blank

**Ant Fall** blank

### Mixed Autonomy Traffic

**Figure Eight v2** blank

**Merge v2** blank

## Citing

To cite this repository in publications, use the following:

```
@misc{h-baselines,
  author = {Kreidieh, Abdul Rahman},
  title = {Hierarchical Baselines},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/AboudyKreidieh/h-baselines}},
}
```

## Bibliography

[1] Dayan, Peter, and Geoffrey E. Hinton. "Feudal reinforcement learning." 
Advances in neural information processing systems. 1993.

[2] Vezhnevets, Alexander Sasha, et al. "Feudal networks for hierarchical 
reinforcement learning." Proceedings of the 34th International Conference on 
Machine Learning-Volume 70. JMLR. org, 2017.

[3] Nachum, Ofir, et al. "Data-efficient hierarchical reinforcement learning."
Advances in Neural Information Processing Systems. 2018.

[4] Levy, Andrew, et al. "Learning Multi-Level Hierarchies with Hindsight." 
(2018).

## Useful Links

The following bullet points contain links developed either by developers of
this repository or external parties that may be of use to individuals
interested in further developing their understanding of hierarchical
reinforcement learning:

* https://thegradient.pub/the-promise-of-hierarchical-reinforcement-learning/
