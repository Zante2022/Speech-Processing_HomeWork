import os

import numpy as np
from hmmlearn import hmm

from process.constant import LABELS, STATES
from speech.extract_mfcc import extract_mfcc, get_mfcc
from sklearn.model_selection import train_test_split

from speech.hmm import  fit_model

if __name__ == '__main__':
    id_sv = 19021289
    fit_model()