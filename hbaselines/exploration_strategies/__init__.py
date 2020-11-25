"""Init file for the exploration strategies."""
# Classic exploration strategies.
from hbaselines.exploration_strategies.classic import EpsilonGreedy
from hbaselines.exploration_strategies.classic import UpperConfidenceBounds
from hbaselines.exploration_strategies.classic import BoltzmannExploration
from hbaselines.exploration_strategies.classic import ThompsonSampling
from hbaselines.exploration_strategies.classic import OutputNoise
from hbaselines.exploration_strategies.classic import ParameterNoise

# Counting-based exploration strategies.
from hbaselines.exploration_strategies.counting import \
    DensityCountingExploration
from hbaselines.exploration_strategies.counting import HashCountingExploration

# Surprise-based exploration strategies.


__all__ = [
    "EpsilonGreedy",
    "UpperConfidenceBounds",
    "BoltzmannExploration",
    "ThompsonSampling",
    "OutputNoise",
    "ParameterNoise",
    "DensityCountingExploration",
    "HashCountingExploration",
]
