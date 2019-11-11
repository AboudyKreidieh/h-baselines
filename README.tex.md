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
  * [TD3](#td3)
  * [Fuly Connected Neural Networks](#fully-connected-neural-networks)
  * [Goal-Conditioned HRL](#hiro-data-efficient-hierarchical-reinforcement-learning)
  * [Meta Period](#meta-period)
  * [Intrinsic Rewards](#intrinsic-rewards)
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

To properly import and run the AntGather environment, you will need to 
first clone and install the `rllab` library. You can do so running the 
following commands:

```
git clone https://github.com/rll/rllab.git
cd rllab
python setup.py develop
```

While all other environments run on all version of MuJoCo, this one will 
require MuJoCo-1.3.1. You may also need to install some missing packages
as well that are required by rllab. If you're installation is 
successful, the following command should not fail:

```
python experiments/run_fcnet.py
```

## Supported Models/Algorithms

This repository currently supports the use several algorithms  of 
goal-conditioned hierarchical reinforcement learning models.

### TD3

We use TD3 as our base policy optimization algorithm. Details on this 
algorithm can be found in the following article: 
https://arxiv.org/pdf/1802.09477.pdf.

To train a policy using this algorithm, create a `TD3` object and 
execute the `learn` method, providing the algorithm the proper policy 
along the process:

```python
from hbaselines.goal_conditioned.algorithm import TD3
from hbaselines.goal_conditioned.policy import FeedForwardPolicy

# create the algorithm object, 
alg = TD3(policy=FeedForwardPolicy, env="AntGather")

# train the policy for the allotted number of timesteps
alg.learn(total_timesteps=1000000)
```

The hyperparameters and modifiable features of this algorithm are as 
follows:

* **policy** (type [ hbaselines.goal_conditioned.policy.ActorCriticPolicy ]) : 
  the policy model to use
* **env** (gym.Env or str) : the environment to learn from (if 
  registered in Gym, can be str)
* **eval_env** (gym.Env or str) : the environment to evaluate from (if 
  registered in Gym, can be str)
* **nb_train_steps** (int) : the number of training steps
* **nb_rollout_steps** (int) : the number of rollout steps
* **nb_eval_episodes** (int) : the number of evaluation episodes
* **actor_update_freq** (int) : number of training steps per actor 
  policy update step. The critic policy is updated every training step.
* **meta_update_freq** (int) : number of training steps per meta policy 
  update step. The actor policy of the meta-policy is further updated at
  the frequency provided by the actor_update_freq variable. Note that 
  this value is only relevant when using the `GoalConditionedPolicy` 
  policy.
* **reward_scale** (float) : the value the reward should be scaled by
* **render** (bool) : enable rendering of the training environment
* **render_eval** (bool) : enable rendering of the evaluation environment
* **verbose** (int) : the verbosity level: 0 none, 1 training 
  information, 2 tensorflow debug
* **policy_kwargs** (dict) : policy-specific hyperparameters

### Fully Connected Neural Networks

We include a generic feed-forward neural network within the repository 
to validate the performance of typically used neural network model on 
the benchmarked environments. This consists of a pair of actor and 
critic fully connected networks with a tanh nonlinearity at the output 
layer of the actor. The output of the actors are also scaled to match 
the desired action space. 

The feed-forward policy can be imported by including the following 
script:

```python
from hbaselines.goal_conditioned.policy import FeedForwardPolicy
```

This model can then be included to the algorithm via the `policy` 
parameter. The input parameters to this policy are as follows:

The modifiable parameters of this policy are as follows:

* **sess** (tf.compat.v1.Session) : the current TensorFlow session
* **ob_space** (gym.space.*) : the observation space of the environment
* **ac_space** (gym.space.*) : the action space of the environment
* **co_space** (gym.space.*) : the context space of the environment
* **buffer_size** (int) : the max number of transitions to store
* **batch_size** (int) : SGD batch size
* **actor_lr** (float) : actor learning rate
* **critic_lr** (float) : critic learning rate
* **verbose** (int) : the verbosity level: 0 none, 1 training 
  information, 2 tensorflow debug
* **tau** (float) : target update rate
* **gamma** (float) : discount factor
* **noise** (float) : scaling term to the range of the action space, 
  that is subsequently used as the standard deviation of Gaussian noise 
  added to the action if `apply_noise` is set to True in `get_action`
* **target_policy_noise** (float) : standard deviation term to the noise
  from the output of the target actor policy. See TD3 paper for more.
* **target_noise_clip** (float) : clipping term for the noise injected 
  in the target actor policy
* **layer_norm** (bool) : enable layer normalisation
* **layers** (list of int) :the size of the Neural network for the policy
* **act_fun** (tf.nn.*) : the activation function to use in the neural 
  network
* **use_huber** (bool) : specifies whether to use the huber distance 
  function as the loss for the critic. If set to False, the mean-squared 
  error metric is used instead

These parameters can be assigned when using the algorithm object by 
assigning them via the `policy_kwargs` term. For example, if you would 
like to train a fully connected network with a hidden size of [64, 64], 
this could be done let so:

```python
from hbaselines.goal_conditioned.algorithm import TD3
from hbaselines.goal_conditioned.policy import FeedForwardPolicy

# create the algorithm object, 
alg = TD3(
    policy=FeedForwardPolicy, 
    env="AntGather",
    policy_kwargs={
        # modify the network to include a hidden shape of [64, 64]
        "layers": [64, 64],
    }
)

# train the policy for the allotted number of timesteps
alg.learn(total_timesteps=1000000)
```

All `policy_kwargs` terms that are not specified are assigned default 
parameters. These default terms are available via the following command:

```python
from hbaselines.goal_conditioned.algorithm import FEEDFORWARD_PARAMS
print(FEEDFORWARD_PARAMS)
```

### Goal-Conditioned HRL

Goal-conditioned HRL models, also known as feudal models, are a variant 
of hierarchical models that have been widely studied in the HRL
community. This repository supports a two-level (Manager/Worker) variant
of this policy, seen in the figure below. The policy can be imported via
the following command:

```python
from hbaselines.goal_conditioned.policy import GoalConditionedPolicy
```

This network consists of a high-level, or Manager, policy $\pi_m$ that 
computes and outputs goals $g_t \sim \pi_m(s_t, c)$ every $k$ time 
steps, and a low-level policy $\pi_w$ that takes as inputs the current 
state and the assigned goals and is encouraged to perform actions 
$a_t \sim \pi_w(s_t, g_t)$ that satisfy these goals via an intrinsic 
reward function: $r_w(s_t, g_t, s_{t+1})$. The contextual term, $c$, 
parametrizes the environmental objective (e.g. desired position to move 
to), and consequently is passed both to the manager policy as well as 
the environmental reward function $r_m(s_t,c)$.

<p align="center"><img src="docs/img/goal-conditioned.png" align="middle" width="50%"/></p>

All of the parameters specified within the 
[Fully Connected Neural Networks](#fully-connected-neural-networks) 
section are valid for this policy as well. Further parameters are 
described in the subsequent sections below.

All `policy_kwargs` terms that are not specified are assigned default 
parameters. These default terms are available via the following command:

```python
from hbaselines.goal_conditioned.algorithm import GOAL_CONDITIONED_PARAMS
print(GOAL_CONDITIONED_PARAMS)
```

### Meta Period

The Manager action period, $k$, can be specified to the policy during 
training by passing the term under the `meta_period` policy parameter. 
This can be assigned through the algorithm as follows:

```python
from hbaselines.goal_conditioned.algorithm import TD3
from hbaselines.goal_conditioned.policy import GoalConditionedPolicy

alg = TD3(
    policy=GoalConditionedPolicy,
    ...,
    policy_kwargs={
        # specify the Manager action period
        "meta_period": 10
    }
)
```

### Intrinsic Rewards

The intrinsic rewards, or $r_w(s_t, g_t, s_{t+1})$, can have a 
significant affect on the training performance of both the Manager and 
Worker policies. Currently, this repository only support one intrinsic 
reward function: negative distance. This is of the form:

$$r_w(s_t, g_t, s_{t+1}) = ||g_t - s_{t+1}||_2$$

if `relative_goals` is set to False, and

$$r_w(s_t, g_t, s_{t+1}) = ||s_t + g_t - s_{t+1}||_2$$

if `relative_goals` is set to True. This attribute is described in the 
next section.

Other intrinsic rewards will be described here once included in the 
repository.

### HIRO (Data Efficient Hierarchical Reinforcement Learning)

TODO

```python
from hbaselines.goal_conditioned.algorithm import TD3
from hbaselines.goal_conditioned.policy import GoalConditionedPolicy

alg = TD3(
    policy=GoalConditionedPolicy,
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
from hbaselines.goal_conditioned.policy import GoalConditionedPolicy

alg = TD3(
    policy=GoalConditionedPolicy,
    ...,
    policy_kwargs={
        # add this line to include HIRO-style off policy corrections
        "off_policy_corrections": True
    }
)
```

### HRL-CG (Inter-Level Cooperation in Hierarchical Reinforcement Learning)

TODO

<p align="center"><img src="docs/img/hrl-cg.png" align="middle" width="90%"/></p>

TODO: describe usage

```python
from hbaselines.goal_conditioned.algorithm import TD3
from hbaselines.goal_conditioned.policy import GoalConditionedPolicy

alg = TD3(
    policy=GoalConditionedPolicy,
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
