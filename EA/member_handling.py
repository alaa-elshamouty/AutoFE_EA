from __future__ import annotations
from sklearn.decomposition import PCA
from typing import Callable, Optional, Tuple, List
import random
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from EA.strategies import Mutation, Recombination, Combiner
from utilities import get_opr_name


class Member:
    """Class to simplify member handling."""
    last_id = 0

    def __init__(
            self,
            initial_x: np.ndarray,
            y_train: np.ndarray,
            model: Callable,
            mutation: Mutation,
            recombination: Recombination,
            trajectory=(None, None, None, None, None),
            fitness_trajectory: List = [],
            seen_operators: set = set()
    ) -> None:
        """
        Parameters
        ----------
        initial_x : np.ndarray
            Initial coordinate of the member

        model : Callable
            The target function that determines the fitness value

        mutation : Mutation
            Hyperparameter that determines which mutation type use

        recombination : Recombination
            Hyperparameter that determines which recombination type to use

        sigma : Optional[float] = None
            Optional hyperparameter that is only active if mutation is gaussian

        recom_prob : Optional[float]
            Optional hyperparameter that is only active if recombination is uniform
        """
        # astype is crucial here. Otherwise numpy might cast everything to int
        self._x = initial_x.astype(float)
        self._y_train = y_train
        self._f_initial = model
        self._f = model
        self._mutation = mutation
        self._recombination = recombination
        self._max_dims = 100

        self._age = 0  # indicates how many offspring were generated from this member
        self._id = Member.last_id
        Member.last_id += 1
        self._x_changed = True
        self._fit_train = 0.0
        self._fit_test = 0.0
        self.traj = trajectory
        self.fitness_traj = fitness_trajectory
        self.seen_oprs = seen_operators

    @property
    def fitness(self) -> float:
        """Retrieve the fitness, recalculating it if x changed"""
        if self._x_changed:
            self._x_changed = False
            # fitting model on the features
            k = 5
            kf = KFold(n_splits=k, random_state=None)
            acc_score = []

            for train_index, test_index in kf.split(self._x):
                X_train, X_test = self._x[train_index, :], self._x[test_index, :]
                y_train, y_test = self._y_train[train_index], self._y_train[test_index]
                self._f.fit(X_train, y_train, overwrite_warning=True)
                pred_values = self._f.predict(X_test)
                acc = accuracy_score(pred_values, y_test)
                acc_score.append(acc)
                self._f = self._f_initial  # restart model to avoid fitting over a fitted model

            self._fit_test = np.average(acc_score)

        return self._fit_test

    @property
    def x_coordinate(self) -> np.ndarray:
        """The current x coordinate"""
        return self._x

    @x_coordinate.setter
    def x_coordinate(self, value: np.ndarray) -> None:
        """Set the new x coordinate"""
        if value.shape[-1] > self._max_dims:
            print(f'Dimension exceeds max dimension, {value.shape[-1]},applying PCA')
            pca = PCA('mle')
            value = pca.fit_transform(value)
        self._x_changed = True
        self._x = value

    def mutate(self) -> Member:
        """Mutation which creates a new offspring

        As a side effect, it will increment the age of this Member.

        Returns
        -------
        Member
            The mutated Member created from this member
        """
        new_x = self.x_coordinate.copy()

        if self._mutation == Mutation.UNIFORM:
            col_id = np.random.randint(new_x.shape[-1])
            opr_info = Combiner.get_random_mutation_opr(self.seen_oprs)
            opr_name = get_opr_name(opr_info[0])
            opr, new_x = Mutation.apply_mutation(opr_info, new_x, y=self._y_train, col_id=col_id)
            trajectory = [(opr_name, self._id, col_id, None, None), self.traj, None]
            self.fitness_traj.append((opr_name, self.fitness))
            self.seen_oprs.add(opr_name)
        elif self._mutation == Mutation.WEIGHTED:
            raise NotImplementedError

        elif self._mutation == Mutation.NONE:
            pass

        else:
            # We won't consider any other mutation types
            raise RuntimeError(f"Unknown mutation {self._mutation}")

        child = Member(
            new_x,
            self._y_train,
            self._f,
            self._mutation,
            self._recombination,
            trajectory,
            self.fitness_traj,
            self.seen_oprs,
        )
        self._age += 1
        return child

    def recombine(self, partner: Member) -> Member:
        """Recombination of this member with a partner

        Parameters
        ----------
        partner : Member
            The other Member to combine with

        Returns
        -------
        Member
            A new Member based on the combination of this one and the partner
        """
        # TODO
        # ----------------
        new_x = self.x_coordinate.copy()
        if self._recombination == Recombination.WEIGHTED:
            raise NotImplementedError
        # ----------------

        elif self._recombination == Recombination.UNIFORM:
            opr_info = Combiner.get_random_crossover_opr()
            opr_name = get_opr_name(opr_info)
            col_id = np.random.randint(new_x.shape[-1])
            partner_col_id = np.random.randint(partner.x_coordinate.shape[-1])
            opr, new_x = Recombination.apply_recombination(opr_info, new_x, col_id, partner.x_coordinate,
                                                           partner_col_id)
            trajectory = [(opr, self._id, col_id, partner._id, partner_col_id), self.traj, partner.traj]
            self.fitness_traj.append((opr_name, self.fitness))
            self.seen_oprs.add(opr_name)
        elif self._recombination == Recombination.NONE:
            # copy is important here to not only get a reference
            new_x = self.x_coordinate.copy()

        else:
            raise NotImplementedError

        # print(f"new point after recombination:\n {new_x.shape}")

        child = Member(
            new_x,
            self._y_train,
            self._f,
            self._mutation,
            self._recombination,
            trajectory,
            self.fitness_traj
        )
        self._age += 1
        return child

    def __str__(self) -> str:
        """Makes the class easily printable"""
        return f"Population member: ID = {self._id}, Age={self._age}, n_cols={self.x_coordinate.shape[-1]}, f(x)={self.fitness}"

    def __repr__(self) -> str:
        """Will also make it printable if it is an entry in a list"""
        return self.__str__() + "\n"
