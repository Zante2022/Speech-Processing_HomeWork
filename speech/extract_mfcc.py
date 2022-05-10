import librosa
import numpy as np


def extract_mfcc(link):
    """
    MFCC feature extraction (39 features, including MFCC, delta, delta_delta)
    :param link: The path to the .wav file
    :return: Numpy.array
    """
    sound, sr = librosa.load(link)
    # print('Load wav file complete')
    mfcc = librosa.feature.mfcc(
        y=sound, n_mfcc=13, sr=sr, n_mels=128, fmax=8000, power=2, n_fft=1024
    )
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    return np.concatenate((mfcc, delta_mfcc, delta2_mfcc))
    # print('Extract feature speech complete')

    return np.concatenate((mfcc, mfcc_delta, mfcc_delta2))


def get_mfcc(link):
    """
    MFCC feature extraction (39 features, including MFCC, delta, delta_delta)
    :param link: The path to the .wav file
    :return: Numpy.array
    """
    sound, sr = librosa.load(link)
    mfcc = librosa.feature.mfcc(y=sound, sr=sr, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    return np.vstack((mfcc, delta_mfcc, delta2_mfcc)).T
    # print('Extract feature speech complete')

