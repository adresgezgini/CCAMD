import json

import numpy as np
import pandas as pd
import sklearn.metrics
import statsmodels.stats.api as sms
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import ParameterGrid
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler

train_data = pd.read_csv('data/training.csv')
test_data = pd.read_csv('data/validation.csv')

X_train = train_data[['speech', 'silence', 'noise', 'music']].astype('float').copy()
X_test = test_data[['speech', 'silence', 'noise', 'music']].astype('float').copy()

malpractice_labels = test_data['malpractice']
y = np.where(malpractice_labels, 1, 0)
pos_label = 1

scaler_zt = StandardScaler()
scaler_pt = PowerTransformer(method='yeo-johnson')

rbm = BernoulliRBM(random_state=0, verbose=0)
kmeans = MiniBatchKMeans(batch_size=4096, random_state=0)
aglo = AgglomerativeClustering(n_clusters=2)

scaler_zt_kmeans = Pipeline([('scaler', scaler_zt), ('kmeans', kmeans)])
scaler_zt_aglo = Pipeline([('scaler', scaler_zt), ('aglo', aglo)])
scaler_pt_kmeans = Pipeline([('scaler', scaler_pt), ('kmeans', kmeans)])
scaler_pt_aglo = Pipeline([('scaler', scaler_pt), ('aglo', aglo)])

scaler_zt_rbm_kmeans = Pipeline([('scaler', scaler_zt), ('rbm', rbm), ('kmeans', kmeans)])
scaler_pt_rbm_kmeans = Pipeline([('scaler', scaler_pt), ('rbm', rbm), ('kmeans', kmeans)])
scaler_zt_rbm_aglo = Pipeline([('scaler', scaler_zt), ('rbm', rbm), ('aglo', aglo)])
scaler_pt_rbm_aglo = Pipeline([('scaler', scaler_pt), ('rbm', rbm), ('aglo', aglo)])

pipelines = {
    'aglo': {
        'plain': aglo,
        'transforms': {
            'zt': scaler_zt_aglo,
            'zt_rbm': scaler_zt_rbm_aglo,
            'pt': scaler_pt_aglo,
            'pt_rbm': scaler_pt_rbm_aglo
        }
    },
    'kmeans': {
        'plain': kmeans,
        'transforms': {
            'zt': scaler_zt_kmeans,
            'zt_rbm': scaler_zt_rbm_kmeans,
            'pt': scaler_pt_kmeans,
            'pt_rbm': scaler_pt_rbm_kmeans
        }
    }
}
# fit the data to the models
for key, value in pipelines.items():
    print(f'fitting the {key} models')
    for key_, value_ in value.items():
        if 'rbm' in key_:
            pass
        else:
            value_.fit(X_train)
            # add the results to the dfs
            train_data[f'{key}_{key_}'] = value_[key].labels_
            # predict the labels of the test set and add the results to the test df
            test_data[f'{key}_{key_}'] = value_.fit_predict(X_test)

param_grid = {
    'rbm__n_components': [2, 20, 45, 70, 135, 170, 200],
    'rbm__learning_rate': np.logspace(-3., 0, 20),
    'rbm__batch_size': [8, 16, 64, 128]
}

# Hyperparameter tuning for rbm

best_F1_zt = 0
best_F1_pt = 0
best_recalls_zt = 0.0
F1s_zt = []
F1s_pt = []
recalls_zt = []
results_rbm = {}

for key, value in pipelines.items():
    for g in ParameterGrid(param_grid):

        model_zt = value['transforms']['zt_rbm']
        model_pt = value['transforms']['pt_rbm']

        model_zt.set_params(**g)
        model_pt.set_params(**g)

        model_zt.fit(X_train)
        model_pt.fit(X_train)

        y_zt = model_zt.fit_predict(X_test)
        y_pt = model_pt.fit_predict(X_test)

        _, count_zt = np.unique(y_zt, return_counts=True)
        _, count_pt = np.unique(y_pt, return_counts=True)

        normal_cluster_zt = np.argmax(count_zt)
        normal_cluster_pt = np.argmax(count_pt)

        y_zt = np.where(y_zt == normal_cluster_zt, 1, 0)
        y_pt = np.where(y_pt == normal_cluster_pt, 1, 0)

        F1_zt = sklearn.metrics.f1_score(
            y, y_zt, pos_label=pos_label)
        F1_pt = sklearn.metrics.f1_score(
            y, y_pt, pos_label=pos_label)

        recall_zt = sklearn.metrics.precision_score(
            y, y_zt, pos_label=pos_label)
        recall_pt = sklearn.metrics.precision_score(
            y, y_pt, pos_label=pos_label)

        F1s_zt.append(F1_zt)
        F1s_pt.append(F1_pt)

        recalls_zt.append(recall_zt)

        # save if best
        if recall_zt > best_recalls_zt:
            best_recalls_zt = recall_zt
            best_F1_zt = F1_zt
            best_grid = g

            print(f'Best Precision: {best_recalls_zt.round(3)}',
                  f'Best F1: {best_F1_zt.round(3)}',
                  json.dumps(best_grid, indent=1), sep='\n')
            print('Confusion Matrices', f'{key}_zt:',
                  f'{sklearn.metrics.confusion_matrix(y, y_zt)}', sep='\n')

        if recall_pt > best_recalls_pt:
            best_recalls_pt = recall_pt
            best_F1_pt = F1_pt
            best_grid = g

            print(f'Best Precision: {best_recalls_zt.round(3)}',
                  f'Best F1: {best_F1_pt.round(3)}',
                  json.dumps(best_grid, indent=1), sep='\n')
            print('Confusion Matrices', f'{key}_pt:',
                  f'{sklearn.metrics.confusion_matrix(y, y_pt)}', sep='\n')
            print('------------------------------------------', '', sep='\n')

    # mean f1 and 95% CI
    f1_mean_zt, ci_f1_zt = sms.DescrStatsW(F1s_zt).tconfint_mean()
    f1_mean_pt, ci_f1_pt = sms.DescrStatsW(F1s_pt).tconfint_mean()

    results_rbm[key] = {
        'F1_mean_ci_zt': [f1_mean_zt, ci_f1_zt],
        'F1_mean_ci_pt': [f1_mean_pt, ci_f1_pt],
        'F1_zt': best_F1_zt,
        'F1_pt': best_F1_pt,
        'recall_zt': best_recalls_zt,
        'recall_pt': best_recalls_pt
    }
