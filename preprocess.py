'''
Command-line script for preprocessing LibriSpeech data for training the VGG
deep feature extractor. It involves, loading the soundfiles, parsing them to
obtain single words extracting features (STFT) and saving them.
'''

import os
import h5py
import numpy as np
import soundfile as sf
from scipy.signal import stft
from glob import glob
from argparse import ArgumentParser
# from libs.data_generator import log_standardize, pad_spec
import matplotlib.pyplot as plt


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
    # data_tmp[:,:,0] = log_standardize(data_tmp[:,:,0])
    data_tmp= np.delete(data_tmp, (128), axis=1)
    a = data_tmp.shape[0]/128
    a = np.int(np.ceil(a))
    padded = np.zeros((a*128, 128, 1))
    padded[:data_tmp.shape[0], :data_tmp.shape[1], :data_tmp.shape[2]] = data_tmp
    padded = padded.reshape((-1, 128, 128, 1))
    return padded

def clips_to_specs(clips, num_random_patches=1):
    clips = remove_short_clips(clips)
    specs = []
    for clip in clips:
        for n in range(num_random_patches):
            specs.append(audio_to_input(clip))
    #specs = np.array([pad_spec(spec) for spec in specs])
    specs = np.array([spec for spec in specs])
    return specs

def remove_short_clips(clips, length=16200):
    good_idxs = []
    for i, clip in enumerate(clips):
        audio = sf.read(clip)
        audio = audio[0]
        if len(audio)>=length:
            good_idxs.append(i)
    clips=np.array(clips)[good_idxs]
    return clips

	
def parse_args():
    parser = ArgumentParser(description='Preprocessing of soundfiles')

    parser.add_argument(
        '-data', '--data',
        type=str, default='./data/LibriSpeech/',
        help='dataset to load'
    )

    parser.add_argument(
        '-dest', '--dest',
        type=str, default='./data/LibriSpeechProcessed/',
        help='destination'
    )

    return parser.parse_args()

# Run script
if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()
    data_path = args.data
    # test_clean = sorted(glob(data_path + "test-clean/*/*/*.flac"))
    # dev_clean = sorted(glob(data_path + "dev-clean/*/*/*.flac"))
    # train_clean = sorted(glob(data_path + "train-clean-100/*/*/*.flac"))
    test_SSN_3 = sorted(glob(data_path + "testSSN/*3.wav"))
    dev_SSN_3 = sorted(glob(data_path + "devSSN/*3.wav"))
    # train_SSN_3 = sorted(glob(data_path + "trainSSN/*3.wav"))
    test_SSN_0 = sorted(glob(data_path + "testSSN/*0.wav"))
    dev_SSN_0 = sorted(glob(data_path + "devSSN/*0.wav"))
    # train_SSN_0 = sorted(glob(data_path + "trainSSN/*0.wav"))

    # specs_test_clean = clips_to_specs(test_clean)
    # np.save(args.dest + 'test_clean', specs_test_clean)
    # specs_dev_clean = clips_to_specs(dev_clean)
    # np.save(args.dest + 'dev_clean', specs_dev_clean)
    # specs_train_clean = clips_to_specs(train_clean)
    # np.save(args.dest + 'train_clean', specs_train_clean)
    specs_test_SSN_3 = clips_to_specs(test_SSN_3)
    np.save(args.dest + 'test_SSN_3', specs_test_SSN_3)
    specs_dev_SSN_3 = clips_to_specs(dev_SSN_3)
    np.save(args.dest + 'dev_SSN_3', specs_dev_SSN_3)
    # specs_train_SSN_3 = clips_to_specs(train_SSN_3)
    # np.save(args.dest + 'train_SSN_3', specs_train_SSN_3)
    specs_test_SSN_0 = clips_to_specs(test_SSN_0)
    np.save(args.dest + 'test_SSN_0', )
    specs_dev_SSN_0 = clips_to_specs(dev_SSN_0)
    np.save(args.dest + 'dev_SSN_0', specs_dev_SSN_0)
    # specs_train_SSN_0 = clips_to_specs(train_SSN_0)
    # np.save(args.dest + 'train_SSN_0', specs_train_SSN_0)

    # file_get = glob(args.dest + "*.npy")
    # savedfile = np.load(file_get[0], allow_pickle=True)
    # print(savedfile[0].shape)

    # splits
    print("Hello!")