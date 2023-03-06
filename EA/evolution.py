from __future__ import annotations

from typing import Callable, List, Tuple
import random

import numpy as np
import torch.cuda
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier
from sklearn import preprocessing
from EA.member_handling import Member
from EA.strategies import Mutation, Recombination, ParentSelection, apply_trajectory


class EA:
    """A class implementing evolutionary algorithm strategies"""

    def __init__(
        self,
        model:Callable,
        initial_X_train:np.ndarray,
        y_train: np.ndarray,
        population_size: int = 10,
        problem_dim: int = 2,
        problem_bounds: Tuple[float, float] = (-1000, 1000), #TODO check bounds of TabPFN
        mutation_type: Mutation = Mutation.UNIFORM,
        recombination_type: Recombination = Recombination.UNIFORM,
        selection_type: ParentSelection = ParentSelection.NEUTRAL,
        sigma: float = 1.0,
        recom_proba: float = 0.5,
        total_number_of_function_evaluations: int = 200,
        children_per_step: int = 5,
        fraction_mutation: float = 0.5,
        nr_of_old_to_kill: int = 1,

    ):
        """
        Parameters
        ----------
        model : Callable
            callable target function we optimize

        population_size: int = 10
            The total population size to use

        problem_dim: int = 2
            The dimension of each member's x

        problem_bounds: Tuple[float, float] = (-30, 30)
            Used to make sure population members are valid

        mutation_type: Mutation = Mutation.UNIFORM
            Hyperparameter to set mutation strategy

        recombination_type: Recombination = Recombination.INTERMEDIATE
            Hyperparameter to set recombination strategy

        selection_type: ParentSelection = ParentSelection.NEUTRAL
            Hyperparameter to set selection strategy

        sigma: float = 1.0
            Conditional hyperparameter dependent on mutation_type GAUSSIAN, defines
            the sigma for the guassian distribution

        recom_proba: float = 0.5
            Conditional hyperparameter dependent on recombination_type UNIFORM.

        total_number_of_function_evaluations: int = 200
            Maximum allowed function evaluations

        children_per_step: int = 5
            How many children to produce per step

        fraction_mutation: float = 0.5
            Balance between sexual and asexual reproduction

        nr_of_old_to_kill: int = 1
            Number of the oldest members to kill for a regularized evolution

        """
        assert 0 <= fraction_mutation <= 1
        assert 0 < children_per_step
        assert 0 < total_number_of_function_evaluations
        assert 0 < sigma
        assert 0 < problem_dim
        assert 0 < population_size


        # Step 1: initialize Population of size `population_size`
        # and then ensure it's sorted by it's fitness
        self.population = [
            Member(
                initial_X_train,
                y_train,
                model,
                problem_bounds,
                mutation_type,
                recombination_type,
                sigma,
                recom_proba,
            )
            for _ in range(population_size)
        ]
        self.population.sort(key=lambda x: x.fitness)

        self.pop_size = population_size
        self.selection = selection_type
        self.max_func_evals = total_number_of_function_evaluations
        self.num_children = children_per_step
        self.frac_mutants = fraction_mutation
        self._func_evals = population_size
        self.nr_old_member_to_kill = nr_of_old_to_kill

        # will store the optimization trajectory and lets you easily observe how
        # often a new best member was generated
        self.trajectory = [self.population[0]]

        print(f"Average fitness of population: {self.get_average_fitness()}")

    def get_average_fitness(self) -> float:
        """The average fitness of the current population"""
        return np.mean([pop.fitness for pop in self.population])

    def select_parents(self) -> List[int]:
        """Method that selects the parents

        Returns
        -------
        List[int]
            The indices of the parents in the sorted population
        """
        parent_ids: List[int] = []

        # TODO
        # ---------------
        if self.selection == ParentSelection.NEUTRAL:
            parent_ids = np.random.permutation(range(self.pop_size))[0:self.num_children].tolist()

        elif self.selection == ParentSelection.FITNESS:
            sum_fitness = sum([m.fitness for m in self.population])
            # print([member.fitness for member in self.population])

            parent_ids = np.random.choice(
                range(self.pop_size - 1, -1, -1),  # xd
                self.num_children, replace=False,
                p=[member.fitness / sum_fitness for member in self.population]
            ).tolist()
        elif self.selection == ParentSelection.TOURNAMENT:
            tournament_size = 3
            for _ in range(self.num_children):
                # Choose random contestants
                fighter_ids = np.random.permutation(range(self.pop_size))[0:tournament_size]
                # fighters = [self.population[i] for i in fighter_ids]
                # FIGHT! Use fact that population is sorted by fitness, so min(fighter_ids) always wins
                parent_ids.append(min(fighter_ids))
        else:
            raise NotImplementedError
        # ---------------

        print(f"Selected parents: {parent_ids}")
        return parent_ids

    def step(self) -> float:
        """Performs one step of the algorithm

        2. Parent selection
        3. Offspring creation
        4. Survival selection

        Returns
        -------
        float
            The average population fitness
        """
        # Step 2: Parent selection
        parent_ids = self.select_parents()

        # Step 3: Variation / create offspring
        # TODO
        # ----------------------
        # for each parent create exactly one offspring, use the frac_mutants
        # parameter to determine if more recombination or mutation should be performed
        children: List[Member] = []
        for id in parent_ids:
            if random.random() < self.frac_mutants:
                # mutation
                children.append(self.population[id].mutate())
            else:
                # recombination
                other_id = random.choice(parent_ids)
                children.append(self.population[id].recombine(self.population[other_id]))
            self._func_evals += 1
        # -----------------------

        print(f"Children: {len(children)}")

        # Step 4: Survival selection
        # (mu + lambda)-selection i.e. combine offspring and parents in one sorted list,
        # keeping the pop_size best of the population
        self.population.extend(children)

        # Resort the population based on Fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)

        # Reduce the population
        self.population = self.population[:self.pop_size]

        # Resort the population based on age
        self.population.sort(key=lambda x: x._age,reverse=True)
        # Kill oldest n members for a regularized evolution
        self.population = self.population[self.nr_old_member_to_kill:]

        # Resort the population based on Fitness again
        self.population.sort(key=lambda x: x.fitness, reverse=True)

        # Append the best Member to the trajectory
        self.trajectory.append(self.population[0])

        return self.get_average_fitness()

    def optimize(self) -> Member:
        """The optimization loop performing the desired amount of function evaluations

        Returns
        -------
        Member
            Returns the best member of the population after optimization
        """
        step = 1
        while self._func_evals < self.max_func_evals:
            avg_fitness = self.step()
            best_fitness = self.population[0].fitness
            lines = [
                "=========",
                f"Step: {step}",
                "=========",
                f"Avg. fitness: {avg_fitness:.7f}",
                f"Best. fitness: {best_fitness:.7f}",
                f"Func evals: {self._func_evals}",
                "----------------------------------",
            ]
            print("\n".join(lines))
            step += 1

        return self.population[0]


if __name__ == "__main__":
    """Simple main to give an example of how to use the EA"""
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    normalizer = preprocessing.Normalizer()
    normalized_train_X = normalizer.fit_transform(X_train)

    # N_ensemble_configurations controls the number of model predictions that are ensembled with feature and class rotations (See our work for details).
    # When N_ensemble_configurations > #features * #classes, no further averaging is applied.
    device = 'cpu' if not torch.cuda.is_available() else 'cuda'
    classifier = TabPFNClassifier(device=device, N_ensemble_configurations=32)

    np.random.seed(0)  # fix seed for comparisons sake

    dimensionality = normalized_train_X.shape[-1] #number of columns
    max_func_evals = 4 #500 * dimensionality
    pop_size = 2
    fraction_mutation = 0.5

    ea = EA(
        model=classifier,
        initial_X_train=normalized_train_X,
        y_train = y_train,
        population_size=pop_size,
        problem_dim=dimensionality,
        mutation_type = Mutation.UNIFORM,
        recombination_type=Recombination.UNIFORM,
        selection_type=ParentSelection.TOURNAMENT,
        total_number_of_function_evaluations=max_func_evals,
        fraction_mutation=fraction_mutation,
    )
    optimum = ea.optimize()
    normalized_test_X = normalizer.transform(X_test)
    new_x_test = apply_trajectory(normalized_test_X, optimum.traj)
    y_eval, p_eval = classifier.predict(new_x_test, return_winning_probability=True)

    # print(ea.trajectory)
    #print(optimum)
    print("#" * 120)
    print(accuracy_score(y_eval,y_test))