import os
import numpy as np
import openml as openml
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def load_openml_dataset(id):
    dataset = openml.datasets.get_dataset(id)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )
    return X,y, categorical_indicator,attribute_names

def load_dataset(dataset_fn=None,id=None):
    if dataset_fn:
        X, y = dataset_fn(return_X_y=True)
    else:
        if not id:
            raise ValueError('id of openml dataset not given')
        X,y,_,_= load_openml_dataset(id)
    return X,y


def get_dataset_split(dataset,save=True):
    if not os.path.exists(f"results/{str(dataset)}"):
        os.mkdir(f"results/{str(dataset)}")
    print('Loadind dataset:{}'.format(dataset))
    if not isinstance(dataset, int):
        X, y = load_dataset(dataset_fn=dataset)
        if X.shape[0] > 10000:
            X = X[:10000, ]

    else:
        X, y = load_dataset(id=dataset)
    print('Splitting Dataset...')
    split = train_test_split(X, y, test_size=0.33, random_state=42)
    if save:
        np.save(f"results/{str(dataset)}/X_train", split[0])
        np.save(f"results/{str(dataset)}/X_test", split[1])
        np.save(f"results/{str(dataset)}/y_train", split[2])
        np.save(f"results/{str(dataset)}/y_test", split[3])

    return split

def normalize_data(dataset, data, normalizer =None, X_train=True, save=True):
    if X_train:
        normalizer = preprocessing.Normalizer()
    name = 'normalized_X_train' if X_train else 'normalized_X_test'
    normalized_data = normalizer.fit_transform(data) if X_train else normalizer.transform(data)
    if save:
        np.save(f"results/{str(dataset)}/{name}", normalized_data)
    return normalizer,normalized_data