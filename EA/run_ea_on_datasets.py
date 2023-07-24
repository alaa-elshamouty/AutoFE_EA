import json
import os
import pickle
import sys

import numpy as np
import torch.cuda
from sklearn.datasets import load_iris
from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier

from BBO.bo import BO
from data.datasets_handling import get_dataset_split

sys.stdout = open('output_log.txt', 'w')
if not os.path.exists('results/'):
    os.mkdir('results/')
results_all = {}

if __name__ == "__main__":
    """Simple main to give an example of how to use the EA given a best found set of hyperparameters for EA"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    classifier = TabPFNClassifier(device=device, N_ensemble_configurations=32)
    np.random.seed(0)  # fix seed for comparison
    dataset_folder = '11'
    dataset = load_iris
    opt_config_dir = '../results_bo/BOHB/{}/opt_cfg.json'.format(dataset_folder)
    X_train, X_test, y_train, y_test = get_dataset_split(dataset, save=True)
    # get the best hyperparameters for EA
    opt_config = json.load(open(opt_config_dir, 'r'))
    bo = BO(dataset=dataset)
    # Run EA with best hyperparameters
    optimum = bo.run_ea(X=X_train, y=y_train, params=opt_config)
    # Evaluate on X_test
    results = bo.evaluate_ea(optimum)
    working_dir = f'../results_debug/{str(dataset)}'
    if not os.path.exists(working_dir):
        os.mkdir(working_dir)
    with open(f'{working_dir}/{str(dataset)}_results.pkl', 'wb') as f:
        pickle.dump(results, f)
