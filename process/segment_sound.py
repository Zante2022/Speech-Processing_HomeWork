import os
import re

from pydub import AudioSegment


def segment_sound(link_folder='data/raw data/19021261_HoangDucHa'):
    """
    separate each word in a sentence
    :param link_folder: path of folder contains record and label of speech
    :return: segment sound word to folder data/segment_data/audio
    """
    os.makedirs('data/segment_data/audio', exist_ok=True)
    id_student = re.findall(r'[0-9]+', link_folder)[0]
    # Files name in folder of member us Audacity record
    file_names = set(file[:-4] for file in os.listdir(link_folder))
    # Sort name
    file_names = sorted(file_names)

    # Process time of sound and label of sound in file .txt
    for file in file_names:
        labels = []
        with open(f"{link_folder}/{file}.txt", "r") as f:
            for line in f:
                if line.strip() == "":
                    continue
                start, stop, label = line.split("\t")
                labels.append([start, stop, label.strip()])
        # export segment label
        for order, label in enumerate(labels):
            audio = AudioSegment.from_wav(f"{link_folder}/{file}.wav")
            audio = audio[int(float(label[0]) * 1000):int(float(label[1]) * 1000)]
            os.makedirs(f"data/segment_data/audio/{id_student}/{label[2]}", exist_ok=True)
            audio.export(
                f"data/segment_data/audio/{id_student}/{label[2]}/{file}_{order+1}.wav", format="wav"
            )
    print(f'Segment {link_folder} done')
