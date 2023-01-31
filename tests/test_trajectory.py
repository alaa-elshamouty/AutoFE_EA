from __future__ import annotations

import numpy as np

from EA.evolution import EA
from EA.strategies import Mutation, Recombination, ParentSelection, apply_trajectory
from tabpfn import TabPFNClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def test_trajectory():



    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    classifier = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)

    #classifier.fit(X_train, y_train)
    #y_eval, p_eval = classifier.predict(X_test, return_winning_probability=True)


    np.random.seed(0)  # fix seed for comparisons sake

    dimensionality = X_train.shape[-1] #number of columns
    max_func_evals = 9 #500 * dimensionality
    pop_size = 3
    fraction_mutation = 0.2

    for selection in ParentSelection:
        ea = EA(
            model=classifier,
            initial_X_train=X_train,
            X_test = X_test,
            y_train = y_train,
            y_test = y_test,
            population_size=pop_size,
            problem_dim=dimensionality,
            mutation_type = Mutation.UNIFORM,
            recombination_type=Recombination.UNIFORM,
            selection_type=selection,
            total_number_of_function_evaluations=max_func_evals,
            fraction_mutation=fraction_mutation,
        )
        optimum = ea.optimize()
        expected = optimum.x_coordinate
        computed = apply_trajectory(X_train,optimum.traj)
        np.allclose(expected,computed,rtol=1e-05, atol=1e-08)

if __name__ == "__main__":
    traj= [(np.delete, 5, 19, None, None), [(np.divide, 0, 23, 0, 6), (None, None, None, None, None), (None, None, None, None, None)], None]
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    new_x_test = apply_trajectory(X_test,traj)
    test_trajectory()