import re

import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer



def check_all(new_x, lower, upper, max_dims):
    new_x = check_values(lower, new_x, upper)
    new_x = check_nan_and_zeros(new_x)
    new_x = check_dims(max_dims, new_x)
    return new_x


def check_dims(max_dims, new_x):
    if new_x.shape[-1] > max_dims:
        print(f'Dimension exceeds max dimension, {new_x.shape[-1]},applying PCA')
        pca = PCA(max_dims - 1)
        new_x = pca.fit_transform(new_x)
    return new_x


def check_nan_and_zeros(new_x):
    if np.any(np.isnan(new_x)):
        print('found NaN values, applying Simple Imputer')
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        new_x = imp_mean.fit_transform(new_x)
    new_x[new_x == 0] = 0.000001
    return new_x


def check_values(lower, new_x, upper):
    if np.any((lower >= new_x) | (new_x >= upper)):
        print(f"Member out of bounds, {new_x.min(), new_x.max()}, applying normalization")
        x = (1 + new_x / (1 + abs(new_x))) * 0.5
        x[np.isnan(x) & (new_x > 0)] = 1
        x[np.isnan(x) & (new_x < 0)] = 0
        new_x = x
    return new_x


def get_opr_name(opr):
    opr_str = str(opr)
    opr_str_clean = re.sub(r'\([^)]*\)', '', opr_str)
    opr_str_clean = re.sub(r'<ufunc \'', '', opr_str_clean)
    opr_str_clean = re.sub(r'\'>', '', opr_str_clean)
    return opr_str_clean
