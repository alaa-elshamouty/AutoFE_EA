import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer


def dim_check(new_x: object, lower: object, upper: object, max_dims: object) -> object:
    if np.any((lower >= new_x) | (new_x >= upper)):
        print(f"Member out of bounds, {new_x.min(),new_x.max()}, applying normalization")
        x = (1 + new_x / (1 + abs(new_x))) * 0.5
        x[np.isnan(x) & (new_x>0)] = 1
        x[np.isnan(x) & (new_x<0)] = 0
        new_x = x    
    if np.any(np.isnan(new_x)):
        print('found NaN values, applying Simple Imputer')
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        new_x = imp_mean.fit_transform(new_x)
    if new_x.shape[-1]>max_dims:
        print(f'Dimension exceeds max dimension, {new_x.shape[-1]},applying PCA')
        pca = PCA(max_dims-1)
        new_x = pca.fit_transform(new_x)
    return new_x

x = np.array([[-np.inf,3,4],[2,3,4]])
new = dim_check(x,lower=0,upper=10,max_dims=10)