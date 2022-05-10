import os
import numpy as np
from hmmlearn import hmm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from process.constant import LABELS, STATES
from speech.extract_mfcc import get_mfcc

import warnings
warnings.filterwarnings("ignore")


def data_model_hmm(label='len'):
    data = []
    for id_sv in os.listdir(f"data/segment_data/audio/"):
        files = os.listdir(f"data/segment_data/audio/{id_sv}/{label}")

        data = data + [get_mfcc(f"data/segment_data/audio/{id_sv}/{label}/{file}") for file in files]

    idx_label = [LABELS.index(label)] * len(data)
    return data, idx_label


def fit_model():
    x_train = {}
    x_test = {}
    y_train = {}
    y_test = {}
    models = {}
    for idx, label in enumerate(LABELS):
        data, idx_label = data_model_hmm(label)
        x_train[label], x_test[label], y_train[label], y_test[label] = \
            train_test_split(data, idx_label, test_size=0.2)

        models[label] = hmm.GMMHMM(n_components=STATES[idx], covariance_type="diag", n_iter=300)
        models[label].fit(X=np.vstack(x_train[label]), lengths=[x.shape[0] for x in x_train[label]])
        print(f"{label}: {len(x_train[label])} / {len(x_test[label])}")
    y_true = []
    y_preds = []
    for label in LABELS:
        for mfcc, target in zip(x_test[label], y_test[label]):
            scores = [models[label].score(mfcc) for label in LABELS]
            preds = np.argmax(scores)
            y_true.append(target)
            y_preds.append(preds)

    report = classification_report(y_true, y_preds)
    print(report)
    return models

