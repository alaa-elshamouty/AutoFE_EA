from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier

from EA.evolution import EA
from EA.strategies import ParentSelection, Mutation, Recombination, apply_trajectory
import numpy as np

if __name__ == "__main__":
    """Simple main to give an example of how to use the EA"""
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # N_ensemble_configurations controls the number of model predictions that are ensembled with feature and class rotations (See our work for details).
    # When N_ensemble_configurations > #features * #classes, no further averaging is applied.

    classifier = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)

    np.random.seed(0)  # fix seed for comparisons sake

    dimensionality = X_train.shape[-1] #number of columns
    max_func_evals = 21 #500 * dimensionality
    pop_size = 2
    fraction_mutation = 0.5

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
        selection_type=ParentSelection.TOURNAMENT,
        total_number_of_function_evaluations=max_func_evals,
        fraction_mutation=fraction_mutation,
    )
    optimum = ea.optimize()
    new_x_test = apply_trajectory(X_test, optimum.traj)
    y_eval, p_eval = classifier.predict(new_x_test, return_winning_probability=True)

    # print(ea.trajectory)
    #print(optimum)
    print("#" * 120)
    print(accuracy_score(y_eval,y_test))
