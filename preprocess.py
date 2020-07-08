'''
Command-line script for preprocessing LibriSpeech data for training the VGG
deep feature extractor. It involves, loading the soundfiles, parsing them to
obtain single words extracting features (STFT) and saving them.
'''

import os

import numpy as np

from argparse import ArgumentParser

import soundfile as sf
from scipy.signal import stft

import h5py


def audio_to_input(audio_file):
    audio = sf.read(audio_file)
    audio = audio[0]
    f, t, seg_stft = stft(audio,
                        window='hamming',
                        nperseg=256,
                        noverlap=128)
    mag_spec = np.abs(seg_stft)
    spec_tmp = np.swapaxes(mag_spec, 0, 1)
    data_tmp = spec_tmp[..., np.newaxis]
    data_tmp[:,:,0] = log_standardize(data_tmp[:,:,0])
    data_tmp= np.delete(data_tmp, (128), axis=1)
    return data_tmp
	
	
def parse_args():
    parser = ArgumentParser(description='Preprocessing of soundfiles')

    parser.add_argument(
        '-data', '--data',
        type=str, default='./LibriSpeech',
        help='dataset to load'
    )

    parser.add_argument(
        '-dest_path', '--dest_path',
        type=str, default='./LibriSpeech_words',
        help='destination of processed data'
    )

    return parser.parse_args()

# Run script
if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()

    # splits
    print("Hello!")