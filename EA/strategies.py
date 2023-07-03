import random
from collections import Counter
from enum import IntEnum
import numpy as np
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA, FastICA, KernelPCA, TruncatedSVD
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.feature_selection import SelectPercentile
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PolynomialFeatures

from utilities import get_opr_name, check_zero, get_inverse_weights


class Combiner:
    one_time_oprs = ['RandomTreesEmbedding',
                     'RBFSampler',
                     'QuantileTransformer',
                     'StandardScaler'
                     ]
    single_ops = [(None, {}),
                  (StandardScaler, {}),
                  (SimpleImputer, {'strategy': 'mean', 'copy': False}),
                  (RandomTreesEmbedding, {}),
                  # (FastICA, {}),
                  (FeatureAgglomeration, {}),
                  # (KernelPCA, {}),
                  (RBFSampler, {}),
                  (SelectPercentile, {'percentile': 90}),
                  # (TruncatedSVD, {}),
                  (QuantileTransformer, {'n_quantiles': 100}),
                  (np.log, {}), (np.square, {}), (np.power, {}), (np.sqrt, {}), (np.abs, {}),
                  # (np.delete, {}),
                  (np.exp, {}), (np.sin, {}), (np.cos, {}), (np.reciprocal, {}),
                  (PCA, {'n_components': 0.8, 'svd_solver': 'full'})]
    single_ops_name = [get_opr_name(opr_info[0]) for opr_info in single_ops]

    @staticmethod
    def get_random_crossover_opr():
        combine_ops = [(PolynomialFeatures, {'degree': 2}), (np.multiply, {}), (np.divide, {})]
        return random.choice(combine_ops)

    @staticmethod
    def get_random_mutation_opr(seen_oprs):
        single_ops_filtered = [opr_info for opr_info in Combiner.single_ops if not (
                get_opr_name(opr_info[0]) in seen_oprs and get_opr_name(opr_info[0]) in Combiner.one_time_oprs)]

        mutation_seen_oprs = [seen_opr for seen_opr in seen_oprs if seen_opr in Combiner.single_ops_name]
        if len(mutation_seen_oprs) > 0:
            counter_seen_oprs = dict(Counter(mutation_seen_oprs))
            weights_seen_oprs = get_inverse_weights(counter_seen_oprs, single_ops_filtered)
            weights = weights_seen_oprs.values()
            index = random.choices(np.arange(len(single_ops_filtered)), weights=weights)[0]
            return single_ops_filtered[index]
        else:
            return random.choice(single_ops_filtered)


class Recombination(IntEnum):
    @staticmethod
    def apply_recombination(opr_info, x, col_id, partner_x, partner_col_id, applying_traj=False):
        new_x = np.float32(x.copy())
        if applying_traj:
            opr_class = opr_info
        else:
            opr_class, params = opr_info
        if opr_class is None:
            return None, new_x
        if 'func' in str(type(opr_class)):
            opr = opr_class
            if 'check' in opr.__name__:
                new_x = opr(new_x)
            else:
                col = x[:, col_id]
                partner_col = partner_x[:, partner_col_id]
                if opr.__name__ == 'divide':
                    partner_col = check_zero(partner_col)
                    new_col = opr(col, partner_col).reshape(-1, 1)
                else:
                    new_col = opr(col, partner_col).reshape(-1, 1)
                new_x = np.hstack((x, new_col))
        else:
            if applying_traj:
                opr = opr_class
                new_x = opr_class.transform(new_x)
            else:
                opr = opr_class(**params)
                new_x = opr.fit_transform(new_x)
        return opr, new_x


class Mutation(IntEnum):
    @staticmethod
    def apply_mutation(opr_info, x, y=None, col_id=None, applying_traj=False):
        new_x = np.float32(x.copy())
        if applying_traj:
            opr_class = opr_info
        else:
            opr_class, params = opr_info
        if opr_class is None:
            return None, new_x

        if 'func' in str(type(opr_class)):
            opr = opr_class
            if 'check' in opr.__name__:
                new_x = opr(new_x)
            else:
                col = new_x[:, col_id].reshape(-1, 1)
                if opr.__name__ == 'delete':
                    new_x = opr(new_x, col_id, axis=1)
                elif opr.__name__ == 'power':
                    if np.all((-10 < col) & (10 > col)):
                        new_x[:, col_id] = opr(col, 3).squeeze()
                    else:
                        print('power 3 would over/underflow, killing operation')
                        return None, x.copy()
                elif opr.__name__ == 'log':  # shifted log
                    abs_col = np.abs(col)
                    new_x[:, col_id] = (np.sign(abs_col) * np.log(1 + np.abs(abs_col))).squeeze()
                elif opr.__name__ == 'sqrt':
                    abs_col = np.abs(col)
                    new_x[:, col_id] = opr(abs_col).squeeze()
                elif opr.__name__ == 'exp':
                    if np.all((-5 < col) & (5 > col)):
                        new_x[:, col_id] = opr(col).squeeze()
                    else:
                        print('exp would over/underflow, killing operation')
                        return None, x.copy()
                elif opr.__name__ == 'reciprocal':
                    col = check_zero(col)
                    new_x[:, col_id] = opr(col).squeeze()
                else:  # square or abs
                    new_x[:, col_id] = opr(col).squeeze()
        else:
            if not applying_traj:
                if 'decomposition' in opr_class.__module__ and len(params) == 0:
                    params['n_components'] = max(1, x.shape[-1] - 1)
                elif 'agg' in opr_class.__module__:
                    params['n_clusters'] = max(x.shape[-1] - 1, 1)
                opr = opr_class(**params)
            else:
                opr = opr_class
            if 'selection' in str(type(opr)):
                new_x = opr.fit_transform(new_x, y) if not applying_traj else opr.transform(new_x)
                if not len(new_x.shape) > 0:
                    print('no features selected, child killed')
                    return None, x.copy()
                elif not new_x.shape[-1] > 0:
                    print('no features selected, child killed')
                    return None, x.copy()
            else:
                new_x = opr.fit_transform(new_x) if not applying_traj else opr.transform(new_x)
                if not isinstance(new_x, np.ndarray):
                    new_x = new_x.toarray()
        return opr, new_x


class ParentSelection(IntEnum):
    """Enum defining the parent selection choice"""

    NEUTRAL = 0
    FITNESS = 1
    TOURNAMENT = 2


def apply_operator(traj, data, partner_data, rec=False):
    new_x = data.copy()
    opr, _, col_id, _, partner_col_id = traj
    if rec:
        _, new_x = Recombination.apply_recombination(opr, new_x, col_id, partner_data, partner_col_id,
                                                     applying_traj=True)
    else:
        _, new_x = Mutation.apply_mutation(opr, new_x, col_id=col_id, applying_traj=True)
    return new_x


def apply_trajectory(dataset, trajectory):
    if trajectory == (None, None, None, None, None):
        return dataset
    traj_current_member = trajectory[0]
    traj_first_member = trajectory[1]
    traj_second_member = trajectory[2]
    if traj_first_member == (None, None, None, None, None) and traj_second_member == (None, None, None, None, None):
        new_x = apply_operator(traj_current_member, dataset, dataset, rec=True)
        return new_x
    elif traj_first_member == (None, None, None, None, None) and not traj_second_member:
        new_x = apply_operator(traj_current_member, dataset, None, rec=False)
        return new_x

    first_member_data = apply_trajectory(dataset, traj_first_member)
    if traj_second_member:
        second_member_data = apply_trajectory(dataset, traj_second_member)
        new_x = apply_operator(traj_current_member, first_member_data, second_member_data, rec=True)
    else:
        new_x = apply_operator(traj_current_member, first_member_data, None, rec=False)

    return new_x


def add_to_trajectory_check_oprs(member_id, oprs, trajectory):
    for opr in oprs:
        trajectory = [(opr, member_id, None, None, None), trajectory, None]
    return trajectory

def weighted_random_selection(seen_oprs=[]):
    while True:
        single_ops_filtered = [opr_info for opr_info in Combiner.single_ops if not (
                get_opr_name(opr_info[0]) in seen_oprs and get_opr_name(opr_info[0]) in Combiner.one_time_oprs)]

        mutation_seen_oprs = [seen_opr for seen_opr in seen_oprs if seen_opr in Combiner.single_ops_name]
        if random.random()<=0.5:
            continue
        if len(mutation_seen_oprs) > 0:
            counter_seen_oprs = dict(Counter(mutation_seen_oprs))
            weights_seen_oprs = get_inverse_weights(counter_seen_oprs, single_ops_filtered)
            weights = weights_seen_oprs.values()
            index = random.choices(np.arange(len(single_ops_filtered)), weights=weights)[0]
            opr = get_opr_name(single_ops_filtered[index][0])
            seen_oprs.append(opr)
        else:
            opr = get_opr_name(random.choice(single_ops_filtered)[0])
            seen_oprs.append(opr)

if __name__ == '__main__':
    weighted_random_selection([])