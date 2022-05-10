import sys

from speech.extract_mfcc import extract_mfcc

if __name__ == '__main__':
    mfcc = extract_mfcc('data/segment_data/audio/19021261/nhay/006_12.wav')
    print(mfcc.shape)