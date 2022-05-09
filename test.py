from process.play_sound import play_sound
from speech.recognition import recognition_by_dtw

if __name__ == '__main__':
    play_sound('data/segment_data/audio/19021261/ban/003_2.wav')
    recognition_by_dtw('data/segment_data/audio/19021261/ban/003_2.wav')
