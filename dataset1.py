import pandas as pd

import dataset

src = dataset.uci_root / 'abalone'
dst = dataset.data_folder / 'dataset1'


if __name__ == '__main__':

    dataset.get(src / 'abalone.data', dst / 'abalone.data')
    dataset.get(src / 'abalone.names', dst / 'abalone.names')

    df = (pd
          .read_csv(dst / 'abalone.data', header=None)
          .pipe(lambda x: x[x[0].isin({'F', 'M'})]))

    X = df[[1, 2, 3, 4, 5, 6, 7]].values
    t = (df[0] == 'M').values
    y = df[8] <= 10

    dataset.save('dataset1', X, t, y)

