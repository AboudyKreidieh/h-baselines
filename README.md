[![Build Status](https://travis-ci.com/AboudyKreidieh/h-baselines.svg?branch=master)](https://travis-ci.com/AboudyKreidieh/h-baselines)
[![Coverage Status](https://coveralls.io/repos/github/AboudyKreidieh/h-baselines/badge.svg?branch=master)](https://coveralls.io/github/AboudyKreidieh/h-baselines?branch=master)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/AboudyKreidieh/h-baselines/blob/master/LICENSE)

# h-baselines

`h-baselines` is a repository of high-performing and benchmarked 
hierarchical reinforcement learning models and algorithms.

The models and algorithms supported within this repository can be found 
[here](#supported-modelsalgorithms), and benchmarking results are 
available [here]().

## Contents

* [Setup Instructions](#setup-instructions)
  * [Basic Installation](#basic-installation)
  * [Installing MuJoCo](#installing-mujoco)
  * [Importing AntGather](#importing-antgather)
* [Supported Models/Algorithms](#supported-modelsalgorithms)
  * [Goal-Conditioned HRL](#hiro-data-efficient-hierarchical-reinforcement-learning)
  * [HIRO (Data Efficient Hierarchical Reinforcement Learning)](#hiro-data-efficient-hierarchical-reinforcement-learning)
  * [HRL-CG (Inter-Level Cooperation in Hierarchical Reinforcement Learning)](#hiro-data-efficient-hierarchical-reinforcement-learning)
* [Environments](#environments)
  * [MuJoCo Environments](#mujoco-environments) <!--  * [Flow Environments](#flow-environments)-->
* [Citing](#citing)
* [Bibliography](#bibliography)
* [Useful Links](#useful-links)

## Setup Instructions

### Basic Installation

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

### Installing MuJoCo

In order to run the MuJoCo environments described within the README, you
will need to install MuJoCo and the mujoco-py package. To install both
components follow the setup instructions located 
[here](https://github.com/openai/mujoco-py). This package should work 
with all versions of MuJoCo (with some changes likely to the version of 
`gym` provided); however, the algorithms have been benchmarked to 
perform well on `mujoco-py==1.50.1.68`.

### Importing AntGather

TODO

## Supported Models/Algorithms

This repository currently supports the use several algorithms  of 
*goal-conditioned hierarchical reinforcement learning* models. We begin
by describing what a goal-conditioned HRL model is, then techniques for 
mitigating the effects of instabilities in training.

### Goal-Conditioned HRL

Goal-conditioned HRL models, also known as feudal models, are a variant 
of hierarchical models that have been widely studied in the HRL
community. This repository supports a two-level (Manager/Worker) variant
of this policy, seen in the figure below. The policy can be imported via
the following command:

```python
from hbaselines.goal_conditioned.policy import GoalConditionedPolicy
```

This network consists of a high-level, or Manager, policy <img src="/tex/be447d665f2aa387ed81a35d066e256b.svg?invert_in_darkmode&sanitize=true" align=middle width=21.03516194999999pt height=14.15524440000002pt/> that 
computes and outputs goals <img src="/tex/2bf33ae14059440820ce394b792cd99e.svg?invert_in_darkmode&sanitize=true" align=middle width=98.10126644999998pt height=24.65753399999998pt/> every <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/> time 
steps, and a low-level policy <img src="/tex/51b5a929b95bcaa91a728fbc3c4eb154.svg?invert_in_darkmode&sanitize=true" align=middle width=19.18963529999999pt height=14.15524440000002pt/> that takes as inputs the current 
state and the assigned goals and is encouraged to perform actions 
<img src="/tex/42d35a15bf7f1d9d42240d4dc81b720d.svg?invert_in_darkmode&sanitize=true" align=middle width=103.61875094999998pt height=24.65753399999998pt/> that satisfy these goals via an intrinsic 
reward function: <img src="/tex/281172fc39903f7b030c2a37e355350d.svg?invert_in_darkmode&sanitize=true" align=middle width=102.71324744999998pt height=24.65753399999998pt/>. The contextual term, <img src="/tex/3e18a4a28fdee1744e5e3f79d13b9ff6.svg?invert_in_darkmode&sanitize=true" align=middle width=7.11380504999999pt height=14.15524440000002pt/>, 
parametrizes the environmental objective (e.g. desired position to move 
to), and consequently is passed both to the manager policy as well as 
the environmental reward function <img src="/tex/8f3686f20d97a88b2ae16496f5e4cc6a.svg?invert_in_darkmode&sanitize=true" align=middle width=60.60137324999998pt height=24.65753399999998pt/>.

<img src="docs/img/goal-conditioned.png" width="50%"/>

### Intrinsic Rewards

TODO

### HIRO (Data Efficient Hierarchical Reinforcement Learning)

TODO

```python
from hbaselines.goal_conditioned.algorithm import TD3

alg = TD3(
    ...,
    policy_kwargs={
        # add this line to include HIRO-style relative goals
        "relative_goals": True
    }
)
```

TODO

```python
from hbaselines.goal_conditioned.algorithm import TD3

alg = TD3(
    ...,
    policy_kwargs={
        # add this line to include HIRO-style off policy corrections
        "off_policy_corrections": True
    }
)
```

### HRL-CG (Inter-Level Cooperation in Hierarchical Reinforcement Learning)

TODO

```python
from hbaselines.goal_conditioned.algorithm import TD3

alg = TD3(
    ...,
    policy_kwargs={
        # add this line to include the connected gradient actor update 
        # procedure to the Manager policy
        "connected_gradients": True,
        # specify the connected gradient (lambda) weight
        "cg_weights": 0.01
    }
)
```

## Environments

We benchmark the performance of all algorithms on a set of standardized 
Mujoco (robotics) and Flow (mixed-autonomy traffic) benchmarks. A 
description of each of the studied environments can be found below.

### MuJoCo Environments

<img src="docs/img/mujoco-envs.png"/>

<!--**Pendulum** -->

<!--This task was initially provided by [5].-->

<!--blank-->

<!--**UR5** -->

<!--This task was initially provided by [5].-->

<!--blank-->

**AntGather**

This task was initially provided by [6].

In this task, a quadrupedal (Ant) agent is placed in a 20x20 space 
with 8 apples and 8 bombs. The agent receives a reward of +1 or 
collecting an apple and -1 for collecting a bomb. All other actions 
yield a reward of 0.

**AntMaze**

This task was initially provided by [3].

In this task, immovable blocks are placed to confine the agent to a
U-shaped corridor. That is, blocks are placed everywhere except at (0,0), (8,0), 
(16,0), (16,8), (16,16), (8,16), and (0,16). The agent is initialized at 
position (0,0) and tasked at reaching a specific target position. "Success" in 
this environment is defined as being within an L2 distance of 5 from the target.

**AntPush**

This task was initially provided by [3].

In this task, immovable blocks are placed every where except at 
(0,0), (-8,0), (-8,8), (0,8), (8,8), (16,8), and (0,16), and a movable block is
placed at (0,8). The agent is initialized at position (0,0), and is tasked with 
the objective of reaching position (0,19). Therefore, the agent must first move 
to the left, push the movable block to the right, and then finally navigate to 
the target. "Success" in this environment is defined as being within an L2 
distance of 5 from the target.

**AntFall**

This task was initially provided by [3].

In this task, the agent is initialized on a platform of height 4. 
Immovable blocks are placed everywhere except at (-8,0), (0,0), (-8,8), (0,8),
(-8,16), (0,16), (-8,24), and (0,24). The raised platform is absent in the 
region [-4,12]x[12,20], and a movable block is placed at (8,8). The agent is 
initialized at position (0,0,4.5), and is with the objective of reaching 
position (0,27,4.5). Therefore, to achieve this, the agent must first push the 
movable block into the chasm and walk on top of it before navigating to the 
target. "Success" in this environment is defined as being within an L2 distance 
of 5 from the target.

<!--### Flow Environments-->

<!--**Figure Eight v2** blank-->

<!--**Merge v2** blank-->

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

[4] TODO: HRL-CG

[5] Levy, Andrew, et al. "Learning Multi-Level Hierarchies with Hindsight." 
(2018).

[6] Florensa, Carlos, Yan Duan, and Pieter Abbeel. "Stochastic neural 
networks for hierarchical reinforcement learning." arXiv preprint 
arXiv:1704.03012 (2017).

## Useful Links

The following bullet points contain links developed either by developers of
this repository or external parties that may be of use to individuals
interested in further developing their understanding of hierarchical
reinforcement learning:

* https://thegradient.pub/the-promise-of-hierarchical-reinforcement-learning/
