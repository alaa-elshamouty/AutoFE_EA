import pickle

from BBO.bo import BO
from data import global_datasets

import argparse

parser = argparse.ArgumentParser(description='AutoFE')
parser.add_argument('--dataset_id', type=int, default=0, help='Dataset id from 0 to 27')
parser.add_argument('--runtime', type=int, default=3600, help='Maximum runtime for BO in seconds')
args = parser.parse_args()
def main(args):
    dataset = global_datasets.datasets[args.dataset_id]
    smac_type = 'BOHB'
    runtime = args.runtime
    working_dir = 'results_bo'
    bo = BO(smac_type=smac_type, runtime=runtime, working_dir=working_dir, dataset=dataset)
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
    main(args)