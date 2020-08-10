from scipy.signal import stft, istft
from glob import glob
import soundfile as sf
import numpy as np
from sklearn.metrics import mean_squared_error
import lws

clean_files = sorted(glob('stft_test_file/*.flac'))

import librosa
print("mag only, hamming, using lws")
mse = np.array([])
lws_processor=lws.lws(256,128, mode="speech")
for file in clean_files:
    audio, _s = sf.read(file)
    audio = np.asarray(audio)
    print("shape of audio", audio.shape)
    Zxx2 = lws_processor.stft(audio)
    _t, _f, Zxx2 = stft(audio, window='hamming', nperseg=256, noverlap=128, padded=True)
    print(Zxx2.shape)
    mag = np.abs(Zxx2)
    # print("mag shape = ",mag.shape)
    # mag = lws_processor.run_lws(mag)
    mag = lws_processor.run_lws(np.transpose(mag))
    print("mag_re shape = ", mag.shape)
    mag = np.transpose(mag)
    # print('mag_re = ',mag)
    _t, audio_recovered = istft(mag, window='hamming', nperseg=256, noverlap=128)
    audio_recovered = np.asarray(audio_recovered[:len(audio)])
    print('shape of audio_recovered = ',audio_recovered.size)
    mse = np.append(mse, mean_squared_error(audio, audio_recovered))
    print(file, 'error=', mse[-1])
    sf.write(file+'_mag_only_hamming_lws.wav', audio_recovered, _s)
print('average mse=', mse.sum() / len(mse))
