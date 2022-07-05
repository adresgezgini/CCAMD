import json

import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MiniBatchKMeans
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler


def f1_recall(y_, y__, pos_label_=1):
    _, count_ = np.unique(y__, return_counts=True)
    normal_cluster_ = np.argmax(count_)
    y__ = np.where(y__ == normal_cluster_, 1, 0)
    f1_ = 1 - sklearn.metrics.f1_score(y_, y__, pos_label=pos_label_)
    recall_ = 1 - sklearn.metrics.precision_score(y_, y__, pos_label=pos_label_)
    return f1_, recall_


train_data = pd.read_csv('/content/CCAMD/data/training.csv')
test_data = pd.read_csv('/content/CCAMD/data/validation.csv')

X_train = train_data[['speech', 'silence', 'noise', 'music']].astype('float').copy()
X_test = test_data[['speech', 'silence', 'noise', 'music']].astype('float').copy()

malpractice_labels = test_data['malpractice']
y = np.where(malpractice_labels, 1, 0)
pos_label = 1
scaler_zt = StandardScaler()
scaler_pt = PowerTransformer(method='yeo-johnson')

n_clusters = 2
rbm = BernoulliRBM(random_state=0, verbose=0)
kmeans = MiniBatchKMeans(batch_size=4096, n_clusters=n_clusters, random_state=0)
aglo = AgglomerativeClustering(n_clusters=n_clusters)

scaler_zt_kmeans = Pipeline([('scaler', scaler_zt), ('kmeans', kmeans)])
scaler_zt_aglo = Pipeline([('scaler', scaler_zt), ('aglo', aglo)])
scaler_pt_kmeans = Pipeline([('scaler', scaler_pt), ('kmeans', kmeans)])
scaler_pt_aglo = Pipeline([('scaler', scaler_pt), ('aglo', aglo)])

scaler_zt_rbm_kmeans = Pipeline([('scaler', scaler_zt), ('rbm', rbm), ('kmeans', kmeans)])
scaler_pt_rbm_kmeans = Pipeline([('scaler', scaler_pt), ('rbm', rbm), ('kmeans', kmeans)])
scaler_zt_rbm_aglo = Pipeline([('scaler', scaler_zt), ('rbm', rbm), ('aglo', aglo)])
scaler_pt_rbm_aglo = Pipeline([('scaler', scaler_pt), ('rbm', rbm), ('aglo', aglo)])

aglo_zt_rbm = {
    'rbm__n_components': 45,
    'rbm__learning_rate': 0.04,
    'rbm__batch_size': 128
}

aglo_pt_rbm = {
    'rbm__n_components': 200,
    'rbm__learning_rate': 0.002,
    'rbm__batch_size': 16
}

kmeans_zt_rbm = {
    'rbm__n_components': 20,
    'rbm__learning_rate': 0.113,
    'rbm__batch_size': 64
}

kmeans_pt_rbm = {
    'rbm__n_components': 2,
    'rbm__learning_rate': 0.004,
    'rbm__batch_size': 8
}

pipelines = {
    'aglo': {
        'plain': aglo,
        'zt': scaler_zt_aglo,
        'zt_rbm': {'model': scaler_zt_rbm_aglo, 'params': aglo_zt_rbm},
        'pt': scaler_pt_aglo,
        'pt_rbm': {'model': scaler_pt_rbm_aglo, 'params': aglo_pt_rbm}
    },
    'kmeans': {
        'plain': kmeans,
        'zt': scaler_zt_kmeans,
        'zt_rbm': {'model': scaler_zt_rbm_kmeans, 'params': kmeans_zt_rbm},
        'pt': scaler_pt_kmeans,
        'pt_rbm': {'model': scaler_pt_rbm_kmeans, 'params': kmeans_pt_rbm}
    }
}
# fit the data to the models
results = {}
for key, value in pipelines.items():
    for key_, value_ in value.items():
        if 'rbm' in key_:
            pass
        else:
            _y = value_.fit_predict(X_test)
            f1, recall = f1_recall(y, _y, pos_label_=1)

            results[f'{key}_{key_}'] = {
                f'F1': f1,
                f'Recall': recall
            }
    with open(f'/content/CCAMD/results/results_{key}.json', 'w') as fp:
        json.dump(results, fp, indent=4)
    results = {}

# rbm fitting
results_rbm = {}

for key, value in pipelines.items():
    model_zt = value['zt_rbm']['model']
    model_pt = value['pt_rbm']['model']

    model_zt.set_params(**value['zt_rbm']['params'])
    model_pt.set_params(**value['pt_rbm']['params'])

    model_zt.fit(X_train)
    model_pt.fit(X_train)

    y_zt = model_zt.fit_predict(X_test)
    y_pt = model_pt.fit_predict(X_test)

    F1_zt, recall_zt = f1_recall(y, y_zt, pos_label_=1)
    F1_pt, recall_pt = f1_recall(y, y_pt, pos_label_=1)

    results_rbm[key] = {
        f'{key}_zt': {
            f'F1': F1_zt,
            f'Recall': recall_zt
        },
        f'{key}_pt': {
            f'F1': F1_pt,
            f'Recall': recall_pt
        }
    }
    with open(f'/content/CCAMD/results/results_rbm_{key}.json', 'w') as fp:
        json.dump(results_rbm[key], fp, indent=4)
