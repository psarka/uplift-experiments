import pandas as pd

import dataset

src = dataset.uci_root / 'adult'
dst = dataset.data_folder / 'dataset2'


if __name__ == '__main__':

    dataset.get(src / 'adult.data', dst / 'adult.data')
    dataset.get(src / 'adult.test', dst / 'adult.test')
    dataset.get(src / 'adult.names', dst / 'adult.names')

    df = (pd.concat([pd.read_csv(dst / 'adult.data', header=None),
                     pd.read_csv(dst / 'adult.test', header=None, skiprows=1)]))

    higher_ed = {' Assoc-acdm',
                 ' Assoc-voc',
                 ' Bachelors',
                 ' Doctorate',
                 ' Masters',
                 ' Some-college'}

    high_income = {' >50K',
                   ' >50K.'}

    X = pd.get_dummies(df[[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]])
    t = df[3].isin(higher_ed)
    y = df[14].isin(high_income)

    dataset.save('dataset2', X, t, y)
