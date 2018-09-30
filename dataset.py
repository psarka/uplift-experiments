from pathlib import Path

import numpy as np
from yarl import URL
import wget


data_folder = Path(__file__).resolve().parent / 'data'
uci_root = URL('https://archive.ics.uci.edu/ml/machine-learning-databases/')


def get(src, dst):

    if not dst.exists():

        if not dst.parent.exists():
            dst.parent.mkdir(parents=True)

        wget.download(str(src), str(dst))


def save(dataset_id, X, t, y):

    np.save(data_folder / dataset_id / 'X.npy', X)
    np.save(data_folder / dataset_id / 't.npy', t)
    np.save(data_folder / dataset_id / 'y.npy', y)


def load(dataset_id):

    return (np.load(data_folder / dataset_id / 'X.npy'),
            np.load(data_folder / dataset_id / 't.npy'),
            np.load(data_folder / dataset_id / 'y.npy'))


def _length(array_or_list):

    return array_or_list.shape[0] if hasattr(array_or_list, 'shape') else len(array_or_list)


def _index(array_or_list, indices):

    if hasattr(array_or_list, 'take'):
        return array_or_list.take(indices, axis=0)

    else:
        return [array_or_list[i] for i in indices]


def shuffled(*arrays, seed=None):

    if len(arrays) == 0:
        return None

    first_length = _length(arrays[0])
    for array in arrays[1:]:
        if _length(array) != first_length:
            raise ValueError(f'Arrays have to be of the same first dimension! '
                             f'{first_length} != {_length(array)}')

    # noinspection PyProtectedMember
    random_state = np.random.mtrand._rand if seed is None else np.random.RandomState(seed)
    # noinspection PyUnresolvedReferences
    indices = np.arange(first_length)
    random_state.shuffle(indices)

    shuffled_arrays = [_index(a, indices) for a in arrays]

    if len(shuffled_arrays) == 1:
        return shuffled_arrays[0]
    else:
        return shuffled_arrays


def train_test_split(*arrays, train_proportion):

    if not 0 <= train_proportion <= 1:
        raise ValueError(f'train_proportion has to be between 0 and 1, got {train_proportion}')

    if len(arrays) == 0:
        return None

    first_length = _length(arrays[0])
    for array in arrays[1:]:
        if _length(array) != first_length:
            raise ValueError(f'Arrays have to be of the same first dimension! '
                             f'{first_length} != {_length(array)}')

    n_train = int(first_length * train_proportion)
    train_arrays = [array[:n_train] for array in arrays]
    test_arrays = [array[n_train:] for array in arrays]

    if len(arrays) == 1:
        return train_arrays[0], test_arrays[0]
    else:
        return train_arrays, test_arrays
