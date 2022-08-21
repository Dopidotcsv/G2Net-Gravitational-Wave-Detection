import numpy as np

def get_data(id_, is_train=True):
    if is_train:
        file = f'data/train/{id_[0]}/{id_[1]}/{id_[2]}/{id_}.npy'
    else:
        file = f'data/test/{id_[0]}/{id_[1]}/{id_[2]}/{id_}.npy'

    return np.load(id_, file)