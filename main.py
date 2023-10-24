import argparse
import json
import os
import pickle

from BBO.bo import BO
from EA.strategies import ParentSelection
from data import global_datasets
from data.datasets_handling import get_dataset_name

parser = argparse.ArgumentParser(description='AutoFE')
parser.add_argument('--dataset_id', type=int, default=0, help='Dataset id from 0 to 27')
parser.add_argument('--runtime', type=int, default=10 * 60 * 60,
                    help='Maximum runtime for BO in seconds. If EA_only set to True then it is maximum number of evaluations')
parser.add_argument('--ea_only', action='store_true', help='Run EA with predefined params')
parser.add_argument('--evaluate', action='store_true', help='Run EA with optimal configurations found')
parser.add_argument('--wandb', action='store_true', help='Activate wandb logging. Requires an account.')
args = parser.parse_args()


def main_ea(args):
    # Run EA with self defined hyperparameters
    dataset = global_datasets.datasets[args.dataset_id]
    bo = BO(dataset=dataset, wandb_logging=args.wandb)
    params = {'population_size': 1,
              'selection_type': ParentSelection.TOURNAMENT,
              'total_number_of_function_evaluations': 200,
              'children_per_step': 1,
              'fraction_mutation': 0.5,
              'max_pop_size': 1,
              'regularizer': 0.5}
    X_train, X_test, y_train, y_test = bo.split
    optimum = bo.run_ea(job_name='ea_only_weight_oprs', X=X_train, y=y_train, X_test=X_test, y_test=y_test,
                        params=params, wandb_logging=args.wandb)

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


def main_evaluate(args):
    # Run EA with best found hyperparameters
    directory = 'results_bo'
    dataset_fn = global_datasets.datasets[args.dataset_id]
    dataset_name = get_dataset_name(dataset_fn)
    opt_config_file = os.path.join(directory, dataset_name, 'opt_cfg.json')
    f = open(opt_config_file)
    params = json.load(f)
    bo = BO(dataset=dataset_fn, wandb_logging=args.wandb)
    X_train, X_test, y_train, y_test = bo.split
    trajectories = {}
    for i in range(3):
        try:
            optimum = bo.run_ea(job_name='opt_cfg', X=X_train, y=y_train, X_test=X_test, y_test=y_test, params=params, wandb_logging=args.wandb)
            trajectories[i] = optimum.traj
            with open(f'{directory}/{dataset_name}/{dataset_name}_trajs.pkl', 'wb') as f:
                pickle.dump(trajectories, f)
        except Exception:
            trajectories[i] = None
            continue
    with open(f'{directory}/{dataset_name}/{dataset_name}_trajs.pkl', 'wb') as f:
        pickle.dump(trajectories, f)


def main(args):
    # Run BO to find the best hyperparameters then run EA with best hyperparameters
    dataset = global_datasets.datasets[args.dataset_id]
    smac_type = 'BOHB'
    runtime = args.runtime
    working_dir = 'results_bo'
    bo = BO(smac_type=smac_type, runtime=runtime, working_dir=working_dir, dataset=dataset, wandb_logging=args.wandb)
    X_train, X_test, y_train, y_test = bo.split
    # get the best hyperparameters for EA
    incumbent = bo.run_bo()
    optimum = bo.run_ea(job_name='optimal_config', X=X_train, y=y_train, params=incumbent, wandb_logging=args.wandb)


if __name__ == "__main__":
    if args.ea_only:
        print('running EA on predefined params')
        main_ea(args)
    elif args.evaluate:
        print('running EA with optimal cfg of parameters found')
        main_evaluate(args)
    else:
        print('running EA with BO')
        main(args)
