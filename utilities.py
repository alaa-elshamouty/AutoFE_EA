import re

import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn import preprocessing


def check_all(new_x, max_dims=100):
    new_x = check_inf_values(new_x)
    new_x, imp_mean = check_nan(new_x)
    new_x = check_zero(new_x)
    new_x, pca = check_dims(max_dims, new_x)
    return new_x, [imp_mean, check_inf_values, check_zero, pca]


def check_dims(max_dims, new_x, pca=None):
    if new_x.shape[-1] > max_dims:
        print(f'Dimension exceeds max dimension, {new_x.shape[-1]},applying PCA')
        if pca is None:
            new_dim = int(max_dims / 2)
            pca = PCA(new_dim)
            new_x = pca.fit_transform(new_x)
        else:
            new_x = pca.transform(new_x)
        print(f'\t new dimension, {new_x.shape[-1]}')
    return new_x, pca


def check_nan(new_x, imp_mean=None):
    if np.any(np.isnan(new_x)):
        print('found NaN values, applying Simple Imputer')
        if imp_mean is None:
            imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
            new_x = imp_mean.fit_transform(new_x)
        else:
            new_x = imp_mean.transform(new_x)
    return new_x, imp_mean


def check_zero(new_x):
    new_x[(-1e-5 < new_x) & (new_x < 0)] = -1e-5
    new_x[(0 < new_x) & (new_x < 1e-5)] = 1e-5
    new_x[new_x == 0] = 1e-5  # to avoid undefined when dividing by zero
    return new_x


def check_inf_values(new_x):
    # Replace outliers with np.NAn
    new_x[new_x > 1e5] = np.nan
    new_x[new_x < -1e5] = np.nan
    return new_x


def get_opr_name(opr):
    opr_str = str(opr)
    opr_str_clean = re.sub(r"<ufunc '(\w+)'>", r'\1', opr_str)
    opr_str_clean = re.sub(r'<function (\w+) at \w+>', r'\1', opr_str_clean)
    opr_str_clean = re.sub(r"<class '.*\.(.*)'>", r'\1', opr_str_clean)
    return opr_str_clean


def get_inverse_weights(counter_seen_oprs, oprs):
    prob_seen_oprs = {}
    inverse_total_sum = np.sum(1 / np.array(list(counter_seen_oprs.values())))
    for opr_info in oprs:
        opr_name = get_opr_name(opr_info[0])
        frequency = counter_seen_oprs[opr_name] if opr_name in counter_seen_oprs else 1e-5
        prob = 1 / (frequency * inverse_total_sum)
        prob_seen_oprs[opr_name] = prob
    return prob_seen_oprs


