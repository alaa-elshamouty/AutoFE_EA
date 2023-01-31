import numpy as np
from sklearn.decomposition import PCA

def dim_check(new_x,lower,upper,max_dims):
    if np.all((lower >= new_x) & (new_x >= upper)):
        print(f"Member out of bounds, {new_x.min(),new_x.max()}, applying normalization")
        norm = np.linalg.norm(new_x, 2)
        new_x /= norm
    if new_x.shape[-1]>max_dims:
        print(f'Dimension exceeds max dimension, {new_x.shape[-1]},applying PCA')
        pca = PCA(max_dims-1)
        new_x = pca.fit_transform(new_x)
    return new_x
