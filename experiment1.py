from pathlib import Path

from frozendict import frozendict
from scipy import stats
from uplift.ensemble import RandomForestClassifier
from uplift.metrics import qini_q

import dataset
from json_store import JsonReader
from json_store import JsonWriter


experiment_folder = Path('experiment') / 'experiment1'
computed_jsons = experiment_folder / 'computed.jsons'


def generator(frozen_rv):
    return lambda seed: frozen_rv.rvs(1, random_state=seed)[0]


def power(base, generator_):
    return lambda seed: base ** generator_(seed)


def integer(generator_):
    return lambda seed: int(generator_(seed))


hyperparameters = {'max_depth': integer(power(10, generator(stats.uniform(0, 3)))),
                   'min_samples_split': integer(power(10, generator(stats.uniform(0, 3)))),
                   'min_samples_leaf': integer(power(10, generator(stats.uniform(0, 3))))}


def parameters_to_compute():

    for dataset_id in ['dataset1']:
        for shuffle_seed in range(9):
            for n_estimators in [100, 1000]:
                for criterion in ['uplift_gini', 'uplift_entropy']:
                    for hp_seed in range(1000):

                        yield {'dataset_id': dataset_id,
                               'shuffle_seed': shuffle_seed,
                               'n_estimators': n_estimators,
                               'criterion': criterion,
                               'hp_seed': hp_seed,
                               **{name: gen(hp_seed) for name, gen in hyperparameters.items()}}


def compute_qini(parameters):

    X_original, t_original, y_original = dataset.load(parameters['dataset_id'])
    X, t, y = dataset.shuffled(X_original, t_original, y_original, seed=parameters['shuffle_seed'])

    ((X_train, t_train, y_train),
     (X_test, t_test, y_test)) = dataset.train_test_split(X, t, y, train_proportion=2/3)

    rfc = RandomForestClassifier(n_estimators=parameters['n_estimators'],
                                 criterion=parameters['criterion'],
                                 max_depth=parameters['max_depth'],
                                 min_samples_split=parameters['min_samples_split'],
                                 min_samples_leaf=parameters['min_samples_leaf'])

    rfc.fit(X_train, y_train, t_train)
    uplift_test = rfc.predict_uplift(X_test)

    return qini_q(y_test, uplift_test, t_test)


if __name__ == '__main__':

    if not experiment_folder.exists():
        experiment_folder.mkdir(parents=True)

    computed = set()
    if computed_jsons.exists():
        with computed_jsons.open() as f:
            for computed_parameters in JsonReader(f):
                del computed_parameters['qini']
                computed.add(frozendict(computed_parameters))

    for ps in parameters_to_compute():
        if frozenset(ps) not in computed:
            qini = compute_qini(ps)
            with computed_jsons.open('a') as f:
                JsonWriter(f).writerow({'qini': qini,
                                        **ps})