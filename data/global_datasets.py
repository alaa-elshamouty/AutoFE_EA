import pandas as pd
from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer

from data.datasets_handling import load_dataset

open_cc_dids = {
    15: 'breast-w',  # crashes,
    29: 'credit-approval',  # nominal crashes
    50: 'tic-tac-toe',  # nominal
    54: 'vehicle',  # nominal
    188: 'eucalyptus',  # nominal crashes
    458: 'analcatdata_authorship',
    469: 'analcatdata_dmft',  # nominal
    1049: ' pc4',
    1050: 'pc3',
    1063: 'kc2',
    1068: 'pc1',
    1510: 'wdbc',
    1494: 'qsar-biodeg',
    1480: 'ilpd',
    1462: 'banknote-authentication',
    1464: 'blood-transfusion-service-center',
    6332: 'cylinder-bands',  # nominal crashed
    23381: 'dresses-sales',  # nominal crashed
    40966: 'MiceProtein',  # has id nominal crashed
    40982: ' steel-plates-fault',
    40994: 'climate-model-simulation-crashes',
    40975: ' car'  # nominal
}
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
open_cc_ids = list(open_cc_dids.keys())
sklearn_dataloaders = [load_iris, load_digits, load_wine]
#datasets = open_ml_ids + open_cc_ids + sklearn_dataloaders
datasets = [458,15,1068,11,22,16,1510,18,31,469,188,37,54,14,23,50]
#datasets = [15,37,23] #23
#datasets = [16]
