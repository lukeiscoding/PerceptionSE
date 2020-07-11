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

def clips_to_features_rand(clips, model, num_random_patches=1):
    specs = []
    for clip in clips:
        for n in range(num_random_patches):
            specs.append(audio_to_input(clip, 16200))
    #specs = np.array([pad_spec(spec) for spec in specs])
    specs = np.array([spec for spec in specs])
    feats = model.predict(specs)
    a, b, c, d = np.shape(feats)
    feats_small = np.zeros((len(clips), b, c, d))
    for i in range(0, len(feats), num_random_patches):
        feat_tmp = np.mean(feats[i:i+num_random_patches], axis=0)
        feats_small[int(i/num_random_patches), ...] = feat_tmp
    return feats_small.reshape(len(clips), -1)

def remove_short_clips(clips, length=16200):
    good_idxs = []
    for i, clip in enumerate(clips):
        audio = sf.read(clip)
        audio = audio[0]
        if len(audio)>=length:
            good_idxs.append(i)
    clips=np.array(noise_clips)[good_idxs]
    return clips

	
def parse_args():
    parser = ArgumentParser(description='Preprocessing of soundfiles')

    parser.add_argument(
        '-data', '--data',
        type=str, default='./data/LibriSpeech/',
        help='dataset to load'
    )

    return parser.parse_args()

# Run script
if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()
    data_path = args.__getattribute__("data")
    test_clean = glob(data_path + "test-clean/*/*/*.flac")
    dev_clean = glob(data_path + "dev-clean/*/*/*.flac")
    train_clean = glob(data_path + "train-clean-100/*/*/*.flac")
    test_SSN_3 = glob(data_path + "testSSN/*3.wav")
    dev_SSN_3 = glob(data_path + "devSSN/*3.wav")
    train_SSN_3 = glob(data_path + "trainSSN/*3.wav")
    test_SSN_0 = glob(data_path + "testSSN/*0.wav")
    dev_SSN_0 = glob(data_path + "devSSN/*0.wav")
    train_SSN_0 = glob(data_path + "trainSSN/*0.wav")
    for tests in test_clean:
        audio = sf.read(tests)
        audio = audio[0]
        f, t, seg_stft = stft(audio,
                              window='hamming',
                              nperseg=256,
                              noverlap=128)
        print(seg_stft)
        mag_spec = np.abs(seg_stft)
        print(mag_spec)
        break



    # splits
    print("Hello!")