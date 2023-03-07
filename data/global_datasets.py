from sklearn.datasets import load_iris, load_digits

open_ml_name_dict = {
    11: 'balance-scale',
    18: 'mfeat-morphological',
    16: 'mfeat-karhunen',
    22: 'mfeat-zernike',
    31: 'credit-g',
    37: 'diabetes',
    23: 'cmc',
    14: 'mfeat-fourier'
}
open_ml_ids = list(open_ml_name_dict.keys())
sklearn_dataloaders = [load_iris, load_digits]
datasets = sklearn_dataloaders + open_ml_ids
