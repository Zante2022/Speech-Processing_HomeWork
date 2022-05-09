import os

import librosa

from process.constant import LABELS, SAMPLE_COUNT, DIR_SEGMENT_DATA
from speech.dtw import dtw
from speech.extract_mfcc import extract_mfcc


def template_mfcc(id_sv=19021289, shuffle=False):
    # create template for speech
    template_mfccs = {}
    for label in LABELS:
        template_mfccs[label] = []
        file_samples = os.listdir(f"data/segment_data/audio/{id_sv}/{label}")
        for file in file_samples[:SAMPLE_COUNT]:
            sample_mfcc = extract_mfcc(f"data/segment_data/audio/{id_sv}/{label}/{file}")
            template_mfccs[label].append(sample_mfcc)
    if shuffle is True:
        for label in LABELS:
            template_mfccs[label] = []
            for k in os.listdir(DIR_SEGMENT_DATA):
                sample_mfcc = extract_mfcc(
                    DIR_SEGMENT_DATA + f"/{k}/{label}/"
                    + os.listdir(DIR_SEGMENT_DATA + f"/{k}/{label}/")[0])
                template_mfccs[label].append(sample_mfcc)
    return template_mfccs


def recognition(file, templates):
    sample_mfcc = extract_mfcc(file)
    distances = {}
    # calculate dtw distance
    for label in LABELS:
        distances[label] = []
        for template in templates[label]:
            distances[label].append(dtw(sample_mfcc, template))

    min_distances = {}
    for label in LABELS:
        min_distances[label] = min(distances[label])

    min_label = min(min_distances, key=min_distances.get)

    return min_label


def recognition_by_dtw(link_wav_file):
    mssv = '19021289'
    template = template_mfcc(id_sv=mssv, shuffle=False)
    recd_label = recognition(link_wav_file, template)

    print('Label is', recd_label)