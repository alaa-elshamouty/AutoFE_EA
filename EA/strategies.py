import random
from enum import IntEnum
from functools import partial
import numpy as np
from autosklearn.pipeline.components.feature_preprocessing import FeaturePreprocessorChoice
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA, FastICA, KernelPCA, TruncatedSVD
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.feature_selection import SelectPercentile
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, normalize, PolynomialFeatures

from utilities import dim_check


class Combiner:
    @staticmethod
    def get_random_mutation_opr():
        single_ops = [(None, {}),
                      (StandardScaler, {}),
                      (SimpleImputer, {'strategy': 'mean', 'copy': False}),
                      (RandomTreesEmbedding, {}),
                      (FastICA, {}),
                      (FeatureAgglomeration, {}),
                      (KernelPCA, {}),
                      (RBFSampler, {}),
                      (SelectPercentile, {'percentile': 50}),
                      (TruncatedSVD, {}),
                      (QuantileTransformer, {}),
                      (np.log, {}), (np.delete, {}), (np.power, {}),
                      (PCA, {})]
        return random.choice(single_ops)

    @staticmethod
    def get_random_crossover_opr():
        combine_ops = [(PolynomialFeatures, {'degree': 2}), (np.multiply, {}), (np.divide, {})]
        return random.choice(combine_ops)


class Recombination(IntEnum):
    """Enum defining the recombination strategy choice"""

    NONE = -1  # can be used when only mutation is required
    UNIFORM = 0  # uniform crossover (only really makes sense for function dimension > 1)
    WEIGHTED = 1  # intermediate recombination

    @staticmethod
    def apply_recombination(opr_info, x, col_id, partner_x, partner_col_id, lower=-np.inf, upper=np.inf, max_dims=100,applying_traj=False):
        new_x = x.copy()
        if applying_traj:
            opr_class=opr_info
        else:
            opr_class, params = opr_info
        if opr_class is None:
            return None, new_x
        if 'func' in str(type(opr_class)):
            opr = opr_class
            col = x[:, col_id]
            partner_col = partner_x[:, partner_col_id]
            if opr.__name__ == 'divide':  # to handle dividing by zero
                new_col = opr(col, partner_col, out=np.zeros_like(col), where=partner_col != 0).reshape(-1, 1)
            else:
                new_col = opr(col, partner_col).reshape(-1, 1)
            new_x = np.hstack((x, new_col))
        else:
            if applying_traj:
                opr=opr_class
                new_x = opr_class.transform(new_x)
            else:
                opr = opr_class(**params)
                new_x = opr.fit_transform(new_x)

        new_x = dim_check(new_x, lower, upper, max_dims=min(x.shape[-1] + 6, max_dims))
        return opr, new_x


class Mutation(IntEnum):
    """Enum defining the mutation strategy choice"""

    NONE = -1  # Can be used when only recombination is required
    UNIFORM = 0  # Uniform mutation
    WEIGHTED = 1  # Gaussian mutation

    @staticmethod
    def apply_mutation(opr_info, x, y=None, col_id=None, lower=-10, upper=10, max_dims=100,
                       applying_traj=False):
        new_x = x.copy()
        if applying_traj:
            opr_class = opr_info
        else:
            opr_class, params = opr_info
        if opr_class is None:
            return None, new_x

        if 'func' in str(type(opr_class)):
            opr = opr_class
            col = new_x[:, col_id].reshape(-1, 1)
            if opr.__name__ == 'delete':
                new_x = opr(new_x, col_id, axis=1)
            elif opr.__name__ == 'power':
                new_x[:, col_id] = opr(col, 2).squeeze()
            else:
                abs_col = np.abs(col)
                new_x[:, col_id] = -1 * opr(np.where(abs_col == 0, 0.00001, abs_col).squeeze())
        else:
            if not applying_traj:
                if 'decomposition' in opr_class.__module__:
                    params['n_components'] = max(x.shape[-1] - 1, x.shape[-1] - 5)
                elif 'agg' in opr_class.__module__:
                    params['n_clusters'] = max(x.shape[-1] - 1, x.shape[-1] - 5)

                opr = opr_class(**params)
            else:
                opr = opr_class
            if 'selection' in str(type(opr)):
                new_x = opr.fit_transform(new_x, y) if not applying_traj else opr.transform(new_x)
            else:
                new_x = opr.fit_transform(new_x) if not applying_traj else opr.transform(new_x)
                if not isinstance(new_x, np.ndarray):
                    new_x = new_x.toarray()

        new_x = dim_check(new_x, lower, upper, min(x.shape[-1] + 6, max_dims))
        return opr, new_x


class ParentSelection(IntEnum):
    """Enum defining the parent selection choice"""

    NEUTRAL = 0
    FITNESS = 1
    TOURNAMENT = 2


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


def apply_operator(traj, data, partner_data, rec=False):
    new_x = data.copy()
    opr, _, col_id, _, partner_col_id = traj
    if rec:
        _, new_x = Recombination.apply_recombination(opr, new_x, col_id, partner_data, partner_col_id,
                                                     applying_traj=True)
    else:
        _, new_x = Mutation.apply_mutation(opr, new_x, col_id=col_id, applying_traj=True)

    return new_x
