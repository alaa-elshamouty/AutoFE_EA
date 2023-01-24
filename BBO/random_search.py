from itertools import product

from BBO.hpo import BBO
from EA.strategies import Mutation, ParentSelection, Recombination


class RandomSearch(BBO):
    def determine_best_hypers(self):
        """Find the best combination with a sweep over the possible hyperparamters.

        # TODO
        Implement either grid or random search to determine the best hyperparameter
        setting of your EA implementation when overfitting to the ackley function.
        The only parameter values you have to consider are:
        * selection_type,
        * mutation_type
        * recombination type.

        You can treat the EA as a black-box by optimizing the black-box-function above.
        Note: the order of your "configuration" has to be as stated below

        Returns
        -------
        (Mutation, ParentSelection, Recombination), float
            Your best trio of strategies and the final fitness value of that strategy
        """
        # TODO
        # ---------------------
        best = (Mutation.NONE, ParentSelection.NEUTRAL, Recombination.NONE)
        fitness = 0.0
        best_perf = float("inf")

        # This is grid search

        for mutation, selection, recombi in product(Mutation, ParentSelection, Recombination):
            fitness = self.evaluate_black_box(mutation, selection, recombi)
            if fitness < best_perf:
                best_perf = fitness
                best = (mutation, selection, recombi)
        # ---------------------

        return best, fitness
