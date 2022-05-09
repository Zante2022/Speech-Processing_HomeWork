import sys

from speech.extract_mfcc import extract_mfcc

if __name__ == '__main__':
    extract_mfcc(sys.argv[1])