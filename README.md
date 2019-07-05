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

**Pendulum** 

blank

**UR5** 

blank

**AntMaze** 

In this task, immovable blocks are placed to confine the agent to a
U-shaped corridor. That is, blocks are placed everywhere except at (0,0), (8,0), 
(16,0), (16,8), (16,16), (8,16), and (0,16). The agent is initialized at 
position (0,0) and tasked at reaching a specific target position. "Success" in 
this environment is defined as being within an L2 distance of 5 from the target.

**AntPush** 

In this task, immovable blocks are placed every where except at 
(0,0), (-8,0), (-8,8), (0,8), (8,8), (16,8), and (0,16), and a movable block is
placed at (0,8). The agent is initialized at position (0,0), and is tasked with 
the objective of reaching position (0,19). Therefore, the agent must first move 
to the left, push the movable block to the right, and then finally navigate to 
the target. "Success" in this environment is defined as being within an L2 
distance of 5 from the target.

**AntFall** 

In this task, the agent is initialized on a platform of height 4. 
Immovable blocks are placed everywhere except at (-8,0), (0,0), (-8,8), (0,8),
(-8,16), (0,16), (-8,24), and (0,24). The raised platform is absent in the 
region [-4,12]x[12,20], and a movable block is placed at (8,8). The agent is 
initialized at position (0,0,4.5), and is with the objective of reaching 
position (0,27,4.5). Therefore, to achieve this, the agent must first push the 
movable block into the chasm and walk on top of it before navigating to the 
target. "Success" in this environment is defined as being within an L2 distance 
of 5 from the target.

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
