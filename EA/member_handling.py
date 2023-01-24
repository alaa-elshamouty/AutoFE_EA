from __future__ import annotations
from sklearn.decomposition import PCA
from typing import Callable, Optional, Tuple
import random
import numpy as np
from sklearn.metrics import accuracy_score

from EA.strategies import Mutation, Recombination, Combiner


class Member:
    """Class to simplify member handling."""

    def __init__(
            self,
            initial_x: np.ndarray,
            X_test: np.ndarray,
            y_train:np.ndarray,
            y_test:np.ndarray,
            model: Callable,
            bounds: Tuple[float, float],
            mutation: Mutation,
            recombination: Recombination,
            sigma: Optional[float] = None,
            recom_prob: Optional[float] = None,
    ) -> None:
        """
        Parameters
        ----------
        initial_x : np.ndarray
            Initial coordinate of the member

        model : Callable
            The target function that determines the fitness value

        bounds : Tuple[float, float]
            Allowed bounds. For simplicities sake we assume that all elements in
            initial_x have the same bounds:
            * bounds[0] lower bound
            * bounds[1] upper bound

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
        self._x_test = X_test
        self._y_train = y_train
        self._y_test = y_test
        self._f_initial = model
        self._f = model
        self._bounds = bounds
        self._mutation = mutation
        self._recombination = recombination
        self._sigma = sigma
        self._recom_prob = recom_prob
        self._max_dims = 100

        # indicates how many offspring were generated from this member
        self._age = 0
        self._x_changed = True
        self._fit_train = 0.0
        self._fit_test = 0.0

    @property
    def fitness(self) -> float:
        """Retrieve the fitness, recalculating it if x changed"""
        if self._x_changed:
            self._x_changed = False
            self._f.fit(self._x,self._y_train)  # fitting model on the features
            test_pred = self._f.predict(self._x_test)
            self._fit_test = accuracy_score(test_pred,self._y_test)
            self._f = self._f_initial  # restart model to avoid fitting over a fitted model
        return self._fit_test

    @property
    def x_coordinate(self) -> np.ndarray:
        """The current x coordinate"""
        return self._x

    @x_coordinate.setter
    def x_coordinate(self, value: np.ndarray) -> None:
        """Set the new x coordinate"""
        lower, upper = self._bounds
        if np.all((lower <= value) & (value <= upper)):
            print(f"Member out of bounds, {value}, applying normalization")
            norm = np.linalg.norm(value, 2)
            value /= norm
        if value.shape[-1]>self._max_dims:
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

        #print(f"new point before mutation:\n {new_x}")

        # TODO modify new_x either through uniform or weighted mutation
        # ---------------
        if self._mutation == Mutation.UNIFORM:
            col_id = np.random.randint(new_x.shape[-1])
            opr = Combiner.single_ops[6]
            #opr = np.random.choice(Combiner.single_ops,1)
            col = new_x[:,col_id].reshape(-1,1)
            if "sklearn" in str(type(opr)):
                new_x[:,col_id] = opr.fit_transform(col).squeeze()
            else:
                if opr.__name__ == 'delete':
                    new_x = opr(new_x,col_id)
                elif opr.__name__ == 'power':
                    new_x[:,col_id] = opr(col,2).squeeze()
                else:
                    new_x[:,col_id] = opr(col)

        elif self._mutation == Mutation.WEIGHTED:
            raise NotImplementedError

        elif self._mutation == Mutation.NONE:
            pass

        else:
            # We won't consider any other mutation types
            raise RuntimeError(f"Unknown mutation {self._mutation}")
        # -----------------

        print(f"new point after mutation: \n {new_x.shape}")
        child = Member(
            new_x,
            self._x_test,
            self._y_train,
            self._y_test,
            self._f,
            self._bounds,
            self._mutation,
            self._recombination,
            self._sigma,
            self._recom_prob,
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

        # TODO
        # ----------------
        elif self._recombination == Recombination.UNIFORM:
            opr = Combiner.combine_ops[1]
            if "sklearn" in str(type(opr)):
                new_x = opr.fit_transform(new_x)
            else:
                col_id = np.random.randint(new_x.shape[-1])
                col = new_x[:,col_id]
                partner_col_id = np.random.randint(partner.x_coordinate.shape[-1])
                partner_col = partner.x_coordinate[:,partner_col_id]
                if opr.__name__ == 'divide': # to handle dividing by zero
                    new_col = opr(col, partner_col, out=np.zeros_like(col), where=partner_col!=0)
                else:
                    new_col = opr(col,partner_col).reshape(-1,1)
                new_x = np.hstack((new_x,new_col))
        # ----------------

        elif self._recombination == Recombination.NONE:
            # copy is important here to not only get a reference
            new_x = self.x_coordinate.copy()

        else:
            raise NotImplementedError

        print(f"new point after recombination:\n {new_x.shape}")
        child = Member(
            new_x,
            self._x_test,
            self._y_train,
            self._y_test,
            self._f,
            self._bounds,
            self._mutation,
            self._recombination,
            self._sigma,
            self._recom_prob,
        )
        self._age += 1
        return child

    def __str__(self) -> str:
        """Makes the class easily printable"""
        return f"Population member: Age={self._age}, n_cols={self.x_coordinate.shape[-1]}, f(x)={self.fitness}"

    def __repr__(self) -> str:
        """Will also make it printable if it is an entry in a list"""
        return self.__str__() + "\n"

