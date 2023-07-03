import pickle

import torch.cuda
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier
from sklearn import preprocessing
import numpy as np

import json
from BBO import bo
from BBO.bo import BO
from EA.evolution import EA
from EA.strategies import Mutation, Recombination, ParentSelection
from data import global_datasets
from data.datasets_handling import load_dataset, get_dataset_split
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

