from abc import abstractmethod

from EA.evolution import EA, Mutation, ParentSelection, Recombination
from target_function import ackley


class BBO:
    def evaluate_black_box(self,
                           mutation: Mutation,
                           selection: ParentSelection,
                           recombination: Recombination,
                           ) -> float:
        """Black-box evaluator of the EA algorithm.

        With your below hpo method you won't have to worry about other parameters

        Parameters
        ----------
        mutation: Mutation
            The choice of mutation strategy

        selection: ParentSelection
            The choice of parent selection strategy

        recombination: Recombination
            The choice of the recombination strategy

        Returns
        -------
        float
            The final fitness after optimizing
        """
        ea = EA(
            model=ackley,
            population_size=20,
            problem_dim=2,
            selection_type=selection,
            total_number_of_function_evaluations=500,
            problem_bounds=(-10, 10),
            mutation_type=mutation,
            recombination_type=recombination,
            sigma=1.0,
            children_per_step=5,
            fraction_mutation=0.5,
            recom_proba=0.5,
        )
        res = ea.optimize()
        return res.fitness

    @abstractmethod
    def determine_best_hypers(self):
        pass
