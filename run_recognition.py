from speech.recognition import recognition_by_dtw, recognition_by_hmm
from playsound import playsound
if __name__ == '__main__':
    link = 'data/segment_data/audio/19021261/ban/003_20.wav'
    playsound(link)
    recognition_by_dtw(link)
    recognition_by_hmm(link)
