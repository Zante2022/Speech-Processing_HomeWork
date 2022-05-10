import librosa


def dtw(sound, reference):
    D, wp = librosa.sequence.dtw(sound, reference, subseq=True, metric="euclidean")
    return D[-1, -1] / wp.shape[0]
