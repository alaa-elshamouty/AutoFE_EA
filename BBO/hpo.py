from abc import abstractmethod

import numpy as np

from sklearn.metrics import accuracy_score

from EA.evolution import EA
from EA.member_handling import Member
from EA.strategies import apply_trajectory
from data.datasets_handling import get_dataset_split, normalize_data
from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier
import torch


class BBO:
    # apply configuration space params
    def __init__(self, dataset, normalizer, save=True):
        self.dataset = dataset
        self.split = get_dataset_split(dataset, save)
        self.results = {}
        self.device = 'cpu' if not torch.cuda.is_available() else 'cuda'
        acc_before_ea = self._evaluate_before_ea()
        self.results['test_acc_before'] = acc_before_ea
        self.normalizer = normalizer

    def _evaluate_before_ea(self):
        X_train, X_test, y_train, y_test = self.split
        classifier = TabPFNClassifier(device=self.device, N_ensemble_configurations=32)
        acc_before_ea = self.fit_run_model(classifier, X_train, y_train, X_test, y_test)
        return acc_before_ea

    def run_ea(self, initial_X, y, params) -> Member:
        model = TabPFNClassifier(device=self.device, N_ensemble_configurations=32)
        ea = EA(model=model, initial_X_train=initial_X, y_train=y, **params)
        res = ea.optimize()
        return res

    def evaluate_ea(self, optimum, save=True):
        _, X_test, y_train, y_test = self.split
        self.results['best_member_fitness'] = optimum.fitness
        best_X_train = optimum.x_coordinate
        self.results['best_member_X_train'] = best_X_train
        if self.normalizer is not None:
            _,X_test = normalize_data(self.dataset, X_test, self.normalizer, X_train=False, save=save)

        trajectory = optimum.traj
        self.results['trajectory_found'] = trajectory
        new_x_test = apply_trajectory(X_test, trajectory)
        if save:
            np.save(f"results/{str(self.dataset)}/X_test_after_trajectory", X_test)

        classifier = TabPFNClassifier(device=self.device, N_ensemble_configurations=32)
        acc_after_ea = self.fit_run_model(classifier, best_X_train, y_train, new_x_test, y_test)
        self.results['test_acc_after'] = acc_after_ea
        return self.results

    @staticmethod
    def fit_run_model(model, X_train, y_train, X_test, y_test):
        model.fit(X_train, y_train, overwrite_warning=True)
        y_eval_before, p_eval_before = model.predict(X_test, return_winning_probability=True)
        test_acc = accuracy_score(y_eval_before, y_test)
        return test_acc

    @abstractmethod
    def _determine_best_hypers(self,config):
        pass
