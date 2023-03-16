import pickle

import torch.cuda
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier
from sklearn import preprocessing
import numpy as np

from EA.evolution import EA
from EA.strategies import Mutation, Recombination, ParentSelection, apply_trajectory
from data import global_datasets
from data.datasets_handling import load_dataset
import sys
import os

sys.stdout = open('output_log.txt', 'w')
if not os.path.exists('results/'):
    os.mkdir('results/')
results_all = {}

if __name__ == "__main__":
    """Simple main to give an example of how to use the EA"""
    # N_ensemble_configurations controls the number of model predictions that are ensembled with feature and class rotations (See our work for details).
    # When N_ensemble_configurations > #features * #classes, no further averaging is applied.
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    classifier = TabPFNClassifier(device=device, N_ensemble_configurations=32)
    np.random.seed(0)  # fix seed for comparison
    for dataset in global_datasets.datasets:
        results_per_dataset = {}
        if not os.path.exists(f"results/{str(dataset)}"):
            os.mkdir(f"results/{str(dataset)}")
        print('#' * 10)
        print('Loadind dataset:{}'.format(dataset))
        if not isinstance(dataset, int):
            X, y = load_dataset(dataset_fn=dataset)
            if X.shape[0] > 10000:
                X = X[:10000, ]

        else:
            X, y = load_dataset(id=dataset)
        print('Splitting Dataset...')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        np.save(f"results/{str(dataset)}/X_train", X_train)
        np.save(f"results/{str(dataset)}/X_test", X_test)
        np.save(f"results/{str(dataset)}/y_train", y_train)
        np.save(f"results/{str(dataset)}/y_test", y_test)
        classifier.fit(X_train, y_train, overwrite_warning=True)
        y_eval_before, p_eval_before = classifier.predict(X_test, return_winning_probability=True)
        acc_before = accuracy_score(y_eval_before, y_test)
        results_per_dataset['test_acc_before'] = acc_before
        normalizer = preprocessing.Normalizer()
        normalized_train_X = normalizer.fit_transform(X_train)
        np.save(f"results/{str(dataset)}/normalized_X_train", X_train)
        print('Setting up EA...')
        dimensionality = normalized_train_X.shape[-1]  # number of columns
        max_func_evals = 3 * dimensionality
        pop_size = 3
        fraction_mutation = 0.7
        children_per_step = 4
        max_pop_size = pop_size * children_per_step
        mutation_type = Mutation.UNIFORM
        recombination_type = Recombination.UNIFORM
        selection_type = ParentSelection.NEUTRAL
        results_per_dataset['EA_params'] = {'dims': dimensionality, 'max_func_eval': max_func_evals,
                                            'pop_size': pop_size, 'frac_mutation': fraction_mutation,
                                            'children_per_step': children_per_step, 'max_pop_size': max_pop_size,
                                            'mutation_type': mutation_type, 'recomb_type': recombination_type,
                                            'selection_type': selection_type}
        ea = EA(
            model=classifier,
            initial_X_train=normalized_train_X,
            y_train=y_train,
            population_size=pop_size,
            mutation_type=mutation_type,
            recombination_type=recombination_type,
            selection_type=selection_type,
            total_number_of_function_evaluations=max_func_evals,
            fraction_mutation=fraction_mutation,
            children_per_step=children_per_step,
            max_pop_size=max_pop_size,
        )

        optimum = ea.optimize()
        print('Finished EA...')

        results_per_dataset['best_member_fitness']= optimum.fitness
        best_X_train = optimum.x_coordinate
        results_per_dataset['best_member_X_train'] = best_X_train
        classifier.fit(best_X_train, y_train, overwrite_warning=True)

        normalized_test_X = normalizer.transform(X_test)
        np.save(f"results/{str(dataset)}/normalized_X_test", X_test)

        trajectory= optimum.traj
        results_per_dataset['trajecotry_found']=trajectory
        new_x_test = apply_trajectory(normalized_test_X, trajectory)
        np.save(f"results/{str(dataset)}/X_test_after_trajectory", X_test)
        print('Evaluating on Test dataset...')
        y_eval, p_eval = classifier.predict(new_x_test, return_winning_probability=True)
        acc_after = accuracy_score(y_eval, y_test)
        results_per_dataset['test_acc_after']=acc_after


        with open(f'results/{str(dataset)}/{str(dataset)}_results.pkl', 'wb') as f:
            pickle.dump(results_per_dataset, f)
        results_all[f'{str(dataset)}']= results_per_dataset

    with open('results/results_all_datasets_1.pkl', 'wb') as f:
        pickle.dump(results_all,f)
