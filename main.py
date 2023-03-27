import pickle

from BBO.bo import BO
from data import global_datasets
import sys
sys.stdout = open('output_log.txt', 'w')
if __name__ == "__main__":
    datasets = global_datasets.datasets
    smac_type = 'BOHB'
    runtime = 21600
    working_dir = 'results_bo'
    normalizer = None
    for dataset in datasets:
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