from itertools import product

from BBO.hpo import BBO
from EA.strategies import Mutation, ParentSelection, Recombination


class RandomSearch(BBO):
    def _determine_best_hypers(self):
        """Find the best combination with a sweep over the possible hyperparamters.
        Returns
        -------
        (Mutation, ParentSelection, Recombination), float
            Your best trio of strategies and the final fitness value of that strategy
        """
        best = (Mutation.NONE, ParentSelection.NEUTRAL, Recombination.NONE)
        fitness = 0.0
        best_perf = float("inf")

        # This is grid search
        for mutation, selection, recombi in product(Mutation, ParentSelection, Recombination):
            fitness = self.evaluate_ea(mutation, selection, recombi)
            if fitness < best_perf:
                best_perf = fitness
                best = (mutation, selection, recombi)
        return best, fitness
