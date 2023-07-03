import os
import numpy as np
import openml as openml
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from utilities import check_all


def load_openml_dataset(id):
    dataset = openml.datasets.get_dataset(id)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )
    return X, y, categorical_indicator, attribute_names


def load_dataset(dataset_fn=None, id=None):
    if dataset_fn:
        X, y = dataset_fn(return_X_y=True)
    else:
        if not id:
            raise ValueError('id of openml dataset not given')
        X, y, _, _ = load_openml_dataset(id)
    return X, y


def get_dataset_split(dataset, dataset_name, save=True):
    if not os.path.exists(f"results/{dataset_name}"):
        os.mkdir(f"results/{dataset_name}")
    print('Loadind dataset:{}'.format(dataset_name))
    if not isinstance(dataset, int):
        X, y = load_dataset(dataset_fn=dataset)
        X = X.astype(float)
        if X.shape[0] > 10000:
            X = X[:10000, ]
    else:
        X, y = load_dataset(id=dataset)
    print('Splitting Dataset...')
    split = train_test_split(X, y, test_size=0.33, random_state=42)

    if save:
        np.save(f"results/{dataset_name}/X_train", split[0])
        np.save(f"results/{dataset_name}/X_test", split[1])
        np.save(f"results/{dataset_name}/y_train", split[2])
        np.save(f"results/{dataset_name}/y_test", split[3])

    return split


def get_dataset_name(dataset_fn):
    try:
        name = dataset_fn.__name__
    except:
        dataset_name = str(dataset_fn)
    else:
        dataset_name = "_".join(name.split("_")[1:])
    return dataset_name
