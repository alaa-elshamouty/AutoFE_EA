import openml as openml
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


