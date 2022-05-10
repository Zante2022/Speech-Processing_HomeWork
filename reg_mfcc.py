import os

from speech.recognition import recognition, template_mfcc
from process.constant import LABELS

if __name__ == '__main__':
    mssv = 19021261
    template = template_mfcc(id_sv=mssv, shuffle=False)
    for label in LABELS:
        correct_label = label
        recd = []

        for file in os.listdir(f"data/segment_data/audio/{mssv}/{label}"):
            if file.endswith(".wav"):
                recd_label = recognition(f"data/segment_data/audio/{mssv}/{label}/{file}", template)
                recd.append(recd_label)

        # calculate accuracy
        accuracy = sum([1 for label in recd if label == correct_label]) / len(recd) * 100

        print(f"Label {correct_label} has accuracy: {accuracy:.2f}%")

