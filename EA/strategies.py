from enum import IntEnum
from functools import partial
import numpy as np
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, normalize, PolynomialFeatures

class Combiner:
    single_ops = [StandardScaler(), QuantileTransformer(), PowerTransformer(), partial(normalize, axis=0), np.log, np.delete, np.power] #Todo remove power transform because TabPFN does it?
    combine_ops = [PolynomialFeatures(2), np.add, np.subtract, np.multiply, np.divide]


class Recombination(IntEnum):
    """Enum defining the recombination strategy choice"""

    NONE = -1  # can be used when only mutation is required
    UNIFORM = 0  # uniform crossover (only really makes sense for function dimension > 1)
    WEIGHTED = 1  # intermediate recombination


class Mutation(IntEnum):
    """Enum defining the mutation strategy choice"""

    NONE = -1  # Can be used when only recombination is required
    UNIFORM = 0  # Uniform mutation
    WEIGHTED = 1  # Gaussian mutation


class ParentSelection(IntEnum):
    """Enum defining the parent selection choice"""

    NEUTRAL = 0
    FITNESS = 1
    TOURNAMENT = 2
