from __future__ import annotations

from typing import Callable, List, Tuple
import random
from copy import deepcopy
import numpy as np

from EA.member_handling import Member
from EA.strategies import Mutation, Recombination, ParentSelection, add_to_trajectory_check_oprs

from tqdm import tqdm
import wandb
from collections import Counter

from utilities import check_all

from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier


class EA:
    """A class implementing evolutionary algorithm strategies"""

    def __init__(
            self,
            job_name,
            dataset_name,
            device: str,
            initial_X_train: np.ndarray,
            y_train: np.ndarray,
            X_test: np.ndarray,
            y_test: np.ndarray,
            population_size: int = 10,
            selection_type: ParentSelection = ParentSelection.NEUTRAL,
            total_number_of_function_evaluations: int = 200,
            children_per_step: int = 3,
            fraction_mutation: float = 0.5,
            max_pop_size: int = 20,
            regularizer: float = 0.75,

    ):
        """
        Parameters
        ----------
        device : Callable
            callable target function we optimize

        population_size: int = 10
            The total population size to use


        mutation_type: Mutation = Mutation.UNIFORM
            Hyperparameter to set mutation strategy

        recombination_type: Recombination = Recombination.INTERMEDIATE
            Hyperparameter to set recombination strategy

        selection_type: ParentSelection = ParentSelection.NEUTRAL
            Hyperparameter to set selection strategy


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
        assert 0 < population_size
        # Step 1: initialize Population of size `population_size`
        # and then ensure it's sorted by it's fitness
        self.X_test = X_test
        self.y_test = y_test
        self.job_name = job_name
        self.population = [
            Member(
                initial_X_train,
                y_train,
                TabPFNClassifier(device=device, N_ensemble_configurations=32),
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
        self.regularizer = regularizer
        # will store the optimization trajectory and lets you easily observe how
        # often a new best member was generated
        self.trajectory = [self.population[0]]
        print(f"Average fitness of population: {self.get_average_fitness()}")
        wandb.init(
            project=f'EA_with_opt_cfg_Final',
            name=dataset_name,
            notes=f'applying EA with optimal configuration found on datasets, final run',
            job_type=self.job_name,
            tags=[dataset_name],
            config={
                'name': dataset_name,
                'population_size': self.pop_size,
                'selection_type': self.selection,
                'max_func_evals': self.max_func_evals,
                'children_per_step': self.num_children,
                'fraction_mutation': self.frac_mutants,
                'ns_every': self.max_pop_size,
                'regularizer': self.regularizer
            }
        )

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
            parents = np.random.choice(self.population, self.num_children)

        elif self.selection == ParentSelection.FITNESS:
            sum_fitness = sum([m.fitness for m in self.population])
            parents = np.random.choice(
                self.population,
                self.num_children, replace=False,
                p=[member.fitness / sum_fitness for member in self.population]
            )
        elif self.selection == ParentSelection.TOURNAMENT:
            tournament_size = len(self.population) // 2 if len(self.population) > 3 else len(self.population)
            parents: List[int] = []
            for _ in range(self.num_children):
                fighters = np.random.choice(self.population, tournament_size, replace=False)
                fitness = [fighter.fitness for fighter in fighters]
                winner_id = np.argmax(fitness)
                parents.append(fighters[winner_id])
        else:
            raise NotImplementedError

        # print(f"Selected parents IDS: {[parent._id for parent in parents]}")
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
                child = parent.mutate()
            else:
                # recombination
                other_parent = random.choice(parents)
                child = parent.recombine(other_parent)

            if child is not None:
                children.append(child)
        self._func_evals += 1
        # -----------------------

        # print(f"Children: {len(children)}")

        # Step 4: Survival selection
        # (mu + lambda)-selection i.e. combine offspring and parents in one sorted list,
        # keeping the pop_size best of the population
        self.population.extend(children)

        # Resort the population based on Fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)

        # if len(self.population) > self.max_pop_size:
        if (self.max_pop_size > 0) and (len(self.population) > self.max_pop_size):
            print('Regularizing :: population number: {}, percentage remove: {}'.format(len(self.population),
                                                                                        self.regularizer))
            keep = int((1 - self.regularizer) * len(self.population))
            self.population = self.population[:keep]

        print('Normal :: population number: {}'.format(len(self.population)))
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
        pbar.set_description(f'EA{self.job_name}')
        step = 1
        best_x_before = self.population[0].x_coordinate
        best_fitness_before = self.population[0].fitness
        test_score_before = self.population[0].evaluate(self.X_test, self.y_test)
        wandb.config['first_best_fitness'] = best_fitness_before
        wandb.config['first_test_score']= test_score_before
        wandb.log({'average fitness': self.get_average_fitness(), 'best fitness': best_fitness_before,
                   'pop_size': len(self.population),
                   'dims_best_member': self.population[0].x_coordinate.shape[-1],
                   'test_score': test_score_before})
        while self._func_evals < self.max_func_evals:
            avg_fitness = self.step()
            best_x_now = self.population[0].x_coordinate
            if best_x_before.shape == best_x_now.shape:
                if np.all(np.equal(best_x_before, best_x_now)):
                    best_member_test = test_score_before
                    best_fitness = best_fitness_before
                else:
                    best_fitness = self.population[0].fitness
                    best_fitness_before = best_fitness
                    best_member_test = self.population[0].evaluate(self.X_test, self.y_test)
                    test_score_before = best_member_test
                    best_x_before = best_x_now
            else:
                best_fitness = self.population[0].fitness
                best_fitness_before = best_fitness
                best_member_test = self.population[0].evaluate(self.X_test, self.y_test)
                test_score_before = best_member_test
                best_x_before = best_x_now
            lines = [
                "=========",
                f"Step: {step}",
                "=========",
                f"Avg. fitness: {avg_fitness:.7f}",
                f"Best. fitness: {best_fitness:.7f}",
                f"Func evals: {self._func_evals}",
                "----------------------------------",
            ]
            wandb.log({'average fitness': avg_fitness, 'best fitness': best_fitness, 'pop_size': len(self.population),
                       'dims_best_member': self.population[0].x_coordinate.shape[-1], 'test_score': best_member_test})
            if step % 20 == 0:
                print("\n".join(lines))
            step += 1
            pbar.update(1)
        operations_counter = Counter(self.population[0].seen_oprs)
        wandb.config['operations'] = dict(operations_counter)
        wandb.config['final_test_score'] = test_score_before
        wandb.config['final_best_fitness'] = best_fitness_before
        pbar.close()
        wandb.finish()
        return self.population[0]
