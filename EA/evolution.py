from __future__ import annotations

from typing import Callable, List, Tuple
import random

import numpy as np
from EA.member_handling import Member
from EA.strategies import Mutation, Recombination, ParentSelection

from tqdm import tqdm


class EA:
    """A class implementing evolutionary algorithm strategies"""

    def __init__(
            self,
            model: Callable,
            initial_X_train: np.ndarray,
            y_train: np.ndarray,
            population_size: int = 10,
            problem_dim: int = 2,
            problem_bounds: Tuple[float, float] = (-1000, 1000),  # TODO check bounds of TabPFN
            mutation_type: Mutation = Mutation.UNIFORM,
            recombination_type: Recombination = Recombination.UNIFORM,
            selection_type: ParentSelection = ParentSelection.NEUTRAL,
            sigma: float = 1.0,
            recom_proba: float = 0.5,
            total_number_of_function_evaluations: int = 200,
            children_per_step: int = 3,
            fraction_mutation: float = 0.5,
            max_pop_size: int = 20,

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

        max_pop_size: int = 20
            Maximum number of population. If exceeded we kill the oldest members to reduce it to population_size

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
        self.max_pop_size = max_pop_size

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
        if self.selection == ParentSelection.NEUTRAL:
            parents = np.random.choice(self.population,self.num_children)

        elif self.selection == ParentSelection.FITNESS:
            sum_fitness = sum([m.fitness for m in self.population])
            parents = np.random.choice(
                self.population,
                self.num_children, replace=False,
                p=[member.fitness / sum_fitness for member in self.population]
            )
        elif self.selection == ParentSelection.TOURNAMENT:
            tournament_size = 3
            parents: List[int] = []
            for _ in range(self.num_children):
                fighters = np.random.choice(self.population,tournament_size,replace=False)
                fitness = [fighter.fitness for fighter in fighters]
                winner_id = np.argmax(fitness)
                parents.append(fighters[winner_id])
        else:
            raise NotImplementedError

        print(f"Selected parents IDS: {[parent._id for parent in parents]}")
        return parents

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
        parents = self.select_parents()

        # Step 3: Variation / create offspring
        children: List[Member] = []
        for parent in parents:
            if random.random() < self.frac_mutants:
                # mutation
                children.append(parent.mutate())
            else:
                # recombination
                other_parent = random.choice(parents)
                children.append(parent.recombine(other_parent))
            self._func_evals += 1
        # -----------------------

        print(f"Children: {len(children)}")

        # Step 4: Survival selection
        # (mu + lambda)-selection i.e. combine offspring and parents in one sorted list,
        # keeping the pop_size best of the population
        self.population.extend(children)

        # Resort the population based on Fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)

        if len(self.population) >= self.max_pop_size:
            # Resort the population based on age
            self.population.sort(key=lambda x: x._age,reverse=True)
            # Reduce the population
            self.population = self.population[self.pop_size:]
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
        pbar = tqdm(total=self.max_func_evals, position=0, leave=True)
        pbar.set_description('EA')
        step = 1
        while self._func_evals < self.max_func_evals:
            before = self._func_evals
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
            after = self._func_evals
            if step % 10 == 0:
                print("\n".join(lines))
            step += 1
            pbar.update(after - before)
        return self.population[0]
