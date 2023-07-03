import os.path
from abc import abstractmethod

import numpy as np
import wandb

from sklearn.metrics import accuracy_score

from EA.evolution import EA
from EA.member_handling import Member
from EA.strategies import apply_trajectory
from data.datasets_handling import get_dataset_split, get_dataset_name
from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier
import torch


class BBO:
    # apply configuration space params
    def __init__(self, dataset, save=True):
        if not os.path.exists('results'):
            os.makedirs('results')
        self.dataset = dataset
        self.dataset_name = get_dataset_name(dataset)
        self.split = get_dataset_split(dataset, self.dataset_name, save)
        self.results = {}
        self.device = 'cpu' if not torch.cuda.is_available() else 'cuda'

    def run_ea(self,job_name,X, y, X_test, y_test, params) -> Member:
        print('applying EA on {}'.format(self.dataset_name))
        ea = EA(job_name,self.dataset_name, device=self.device, initial_X_train=X, y_train=y, X_test=X_test,
                y_test=y_test, **params)
        res = ea.optimize()
        return res

    def evaluate_ea(self, optimum, save=True):
        X_train, X_test, y_train, y_test = self.split
        classifier = TabPFNClassifier(device=self.device, N_ensemble_configurations=32)

        # Before Feature Engineering
        test_acc_before_ea = self.fit_run_model(classifier, X_train, y_train, X_test, y_test)
        self.results['test_acc_before'] = test_acc_before_ea

        # After Feature Engineering
        self.results['best_member_fitness'] = optimum.fitness
        self.results['fitness_trajectory'] = optimum.fitness_traj
        best_X_train = optimum.x_coordinate

        trajectory = optimum.traj
        self.results['trajectory_found'] = trajectory
        new_x_train = apply_trajectory(X_train, trajectory)
        new_x_test = apply_trajectory(X_test, trajectory)
        if save:
            np.save(f"results/{str(self.dataset)}/X_train_best_member", best_X_train)
            np.save(f"results/{str(self.dataset)}/X_train_after_trajectory", new_x_train)
            np.save(f"results/{str(self.dataset)}/X_test_after_trajectory", new_x_test)

        # Reset Model and then test on new features
        classifier = TabPFNClassifier(device=self.device, N_ensemble_configurations=32)
        test_acc_after_ea = self.fit_run_model(classifier, new_x_train, y_train, new_x_test, y_test)
        self.results['test_acc_after'] = test_acc_after_ea
        return self.results

    @staticmethod
    def fit_run_model(model, X_train, y_train, X_test, y_test):
        model.fit(X_train, y_train, overwrite_warning=True)
        y_eval_test, p_eval_test = model.predict(X_test, return_winning_probability=True)
        test_acc = accuracy_score(y_eval_test, y_test)
        return test_acc

    @staticmethod
    def run_model(model, X, y):
        y_eval, p_eval_before = model.predict(X, return_winning_probability=True)
        acc = accuracy_score(y_eval, y)
        return acc

    @abstractmethod
    def _determine_best_hypers(self, config):
        pass
