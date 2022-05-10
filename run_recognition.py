from speech.recognition import recognition_by_dtw, recognition_by_hmm
from playsound import playsound
if __name__ == '__main__':
    link = 'data/segment_data/audio/19021285/xuong/007_6.wav'
    playsound(link)
    recognition_by_dtw(link)
    recognition_by_hmm(link)
