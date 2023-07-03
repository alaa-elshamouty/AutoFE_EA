import os
import pickle
from collections import OrderedDict

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from BBO.bo import BO
from EA.strategies import Recombination, Mutation
from data import global_datasets
import argparse
from data.datasets_handling import get_dataset_name
from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier

from utilities import get_opr_name
from tqdm import tqdm
from visualiser.visualiser import Visualiser as vs

parser = argparse.ArgumentParser(description='AutoFE')
parser.add_argument('--dataset_id', type=int, default=0, help='Dataset id from 0 to 27')
args = parser.parse_args()


def apply_operator(traj, data, partner_data, rec=False):
    new_x = data.copy()
    opr, _, col_id, _, partner_col_id = traj
    if rec:
        _, new_x = Recombination.apply_recombination(opr, new_x, col_id, partner_data, partner_col_id,
                                                     applying_traj=True)
    else:
        _, new_x = Mutation.apply_mutation(opr, new_x, col_id=col_id, applying_traj=True)
    return get_opr_name(opr), new_x


@vs(node_properties_kwargs={"shape": "record", "color": "#f57542", "style": "filled", "fillcolor": "grey"})
def apply_trajectory_simultaneuosly(dataset_train, dataset_test, y_train, y_test, bo, classifier, trajectory, opr_names,
                                    train_scores, test_scores):
    if trajectory == (None, None, None, None, None):
        return dataset_train, dataset_test, opr_names, train_scores, test_scores
    traj_current_member = trajectory[0]
    traj_first_member = trajectory[1]
    traj_second_member = trajectory[2]
    if traj_first_member == (None, None, None, None, None) and traj_second_member == (None, None, None, None, None):
        opr_name, new_x_train = apply_operator(traj_current_member, dataset_train, dataset_train, rec=True)
        _, new_x_test = apply_operator(traj_current_member, dataset_test, dataset_test, rec=True)
        try:
            train_acc = evaluate_train(classifier, new_x_train, y_train)
            test_acc = bo.fit_run_model(classifier, new_x_train, y_train, new_x_test, y_test)
            opr_names.append(f'{opr_name}_{traj_current_member[1]}_{traj_current_member[2]}')
            train_scores.append(train_acc)
            test_scores.append(test_acc)
        except Exception:
            pass
        return new_x_train, new_x_test, opr_names, train_scores, test_scores
    elif traj_first_member == (None, None, None, None, None) and not traj_second_member:
        opr_name, new_x_train = apply_operator(traj_current_member, dataset_train, None, rec=False)
        _, new_x_test = apply_operator(traj_current_member, dataset_test, None, rec=False)
        try:
            train_acc = evaluate_train(classifier, new_x_train, y_train)
            test_acc = bo.fit_run_model(classifier, new_x_train, y_train, new_x_test, y_test)
            opr_names.append(f'{opr_name}_{traj_current_member[1]}_{traj_current_member[2]}')
            train_scores.append(train_acc)
            test_scores.append(test_acc)
        except Exception:
            pass
        return new_x_train, new_x_test, opr_names, train_scores, test_scores

    first_member_data_train, first_member_data_test, opr_names, train_scores, test_scores = apply_trajectory_simultaneuosly(
        dataset_train=dataset_train, dataset_test=dataset_test, y_train=y_train, y_test=y_test, bo=bo,
        classifier=classifier, trajectory=traj_first_member, opr_names=opr_names, train_scores=train_scores,
        test_scores=test_scores)
    if traj_second_member:
        second_member_data_train, second_member_data_test, opr_names, train_scores, test_scores = apply_trajectory_simultaneuosly(
            dataset_train=dataset_train, dataset_test=dataset_test, y_train=y_train, y_test=y_test, bo=bo,
            classifier=classifier, trajectory=traj_second_member, opr_names=opr_names, train_scores=train_scores,
            test_scores=test_scores)
        opr_name, new_x_train = apply_operator(traj_current_member, first_member_data_train, second_member_data_train,
                                               rec=True)
        _, new_x_test = apply_operator(traj_current_member, first_member_data_test, second_member_data_test, rec=True)
    else:
        opr_name, new_x_train = apply_operator(traj_current_member, first_member_data_train, None, rec=False)
        _, new_x_test = apply_operator(traj_current_member, first_member_data_test, None, rec=False)
    try:
        train_acc = evaluate_train(classifier, new_x_train, y_train)
        test_acc = bo.fit_run_model(classifier, new_x_train, y_train, new_x_test, y_test)
        opr_names.append(f'{opr_name}_{traj_current_member[1]}_{traj_current_member[2]}')
        train_scores.append(train_acc)
        test_scores.append(test_acc)
    except Exception:
        pass
    return new_x_train, new_x_test, opr_names, train_scores, test_scores


def evaluate_train(classifier, X, y):
    k = 5
    kf = KFold(n_splits=k, random_state=None)
    acc_score = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index, :], X[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        classifier.fit(X_train, y_train, overwrite_warning=True)
        pred_values = classifier.predict(X_test)
        acc = accuracy_score(pred_values, y_test)
        acc_score.append(acc)

    return np.average(acc_score)


def main():
    directory = '../results_bo'
    for j in range(3):
        dataset_fn = global_datasets.datasets[j]
        dataset_name = get_dataset_name(dataset_fn)
        trajectory_file = os.path.join(directory, dataset_name, f'{dataset_name}_trajs.pkl')
        try:
            with open(trajectory_file, 'rb') as f:
                trajectory = pickle.load(f)
            bo = BO(dataset=dataset_fn)
            X_train, X_test, y_train, y_test = bo.split
            classifier = TabPFNClassifier(device=bo.device, N_ensemble_configurations=32)
            data = {}
            for i in tqdm(range(len(trajectory))):
                # Before Feature Engineering
                train_acc_before = evaluate_train(classifier, X_train, y_train)
                test_acc_before = bo.fit_run_model(classifier, X_train, y_train, X_test, y_test)

                # After Feature Engineering
                opr_names = []
                train_scores = []
                test_scores = []
                opr_names.append('RAW')
                train_scores.append(train_acc_before)
                test_scores.append(test_acc_before)
                new_x_train, new_x_test, opr_names, train_scores, test_scores = apply_trajectory_simultaneuosly(X_train,
                                                                                                                X_test,
                                                                                                                y_train,
                                                                                                                y_test,
                                                                                                                bo,
                                                                                                                classifier,
                                                                                                                trajectory[
                                                                                                                    i],
                                                                                                                opr_names,
                                                                                                                train_scores,
                                                                                                                test_scores)
                info = list(OrderedDict.fromkeys(zip(opr_names,train_scores,test_scores)))
                data[i] = {'opr_names': [opr_name for opr_name,_,_ in info],
                           'train_scores': [train_score for _,train_score,_ in info],
                           'test_scores': [test_score for _,_,test_score in info]}
                with open(f'{directory}/{dataset_name}/{dataset_name}_evaluation_traj_new.pkl', 'wb') as f:
                    pickle.dump(data, f)
        except Exception:
            continue


if __name__ == '__main__':
    main()
