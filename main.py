import os
import pickle

from BBO.bo import BO
from EA.strategies import Mutation, Recombination, ParentSelection
from data import global_datasets

import argparse

parser = argparse.ArgumentParser(description='AutoFE')
parser.add_argument('--dataset_id', type=int, default=0, help='Dataset id from 0 to 27')
parser.add_argument('--job_name', type=int, help='Job name')
parser.add_argument('--runtime', type=int, default=10 * 60 * 60,
                    help='Maximum runtime for BO in seconds. If EA_only set to True then it is maximum number of evaluations')
parser.add_argument('--ea_only', action='store_true', help='Run EA with predefined params')
args = parser.parse_args()

job_name = args.job_name


def main_ea(args):
    dataset = global_datasets.datasets[args.dataset_id]
    runtime = args.runtime
    bo = BO(job_name=job_name,dataset=dataset)
    # Run EA with self defined hyperparameters
    params = {'population_size': 5,
              'mutation_type': Mutation.UNIFORM,
              'recombination_type': Recombination.UNIFORM,
              'selection_type': ParentSelection.TOURNAMENT,
              'total_number_of_function_evaluations': runtime,
              'children_per_step': 1,
              'fraction_mutation': 0.5,
              'max_pop_size': 100,
              'normalize': True,
              'regularizer': 0.25}
    X_train, X_test, y_train, y_test = bo.split
    optimum = bo.run_ea(X=X_train, y=y_train, params=params)
    # Evaluate on X_test
    results = bo.evaluate_ea(optimum)
    working_dir = 'results_ea_' + str(args.job_name)
    if not os.path.exists(working_dir):
        os.mkdir(working_dir)
    dataset_dir = os.path.join(working_dir, str(dataset))
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    with open(f'{dataset_dir}/results.pkl', 'wb') as f:
        pickle.dump(results, f)


def main(args):
    dataset = global_datasets.datasets[args.dataset_id]
    smac_type = 'BOHB'
    runtime = args.runtime
    working_dir = 'results_bo'
    bo = BO(job_name=job_name,smac_type=smac_type, runtime=runtime, working_dir=working_dir, dataset=dataset)
    X_train, X_test, y_train, y_test = bo.split
    # get the best hyperparameters for EA
    incumbent = bo.run_bo()
    # Run EA with best hyperparameters
    optimum = bo.run_ea(X=X_train, y=y_train, params=incumbent)
    # Evaluate on X_test
    results = bo.evaluate_ea(optimum)
    with open(f'results/{str(dataset)}/{str(dataset)}_results.pkl', 'wb') as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    if args.ea_only:
        print('running EA on predefined params')
        main_ea(args)
    else:
        print('running EA with BO')
        main(args)
