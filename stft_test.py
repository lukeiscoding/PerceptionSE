from scipy.signal import stft, istft
from glob import glob
import soundfile as sf
import numpy as np
from sklearn.metrics import mean_squared_error

clean_files = sorted(glob('stft_test_file/*.flac'))

# test for stft and istft
print("full complex number")
for file in clean_files:
    audio, _s = sf.read(file)
    audio = np.asarray(audio)
    _t, _f, Zxx = stft(audio, window='hamming', nperseg=256, noverlap=128, padded=True)
    _t, audio_recovered = istft(Zxx, window='hamming', nperseg=256, noverlap=128)
    audio_recovered = np.asarray(audio_recovered[:len(audio)])
    assert audio.all() == audio_recovered.all()
    print(file, "lossless")

# test for mag only, hamming
print("mag only, hamming")
mse = np.array([])
for file in clean_files:
    audio, _s = sf.read(file)
    audio = np.asarray(audio)
    _t, _f, Zxx = stft(audio, window='hamming', nperseg=256, noverlap=128, padded=True)
    _t, audio_recovered = istft(np.abs(Zxx), window='hamming', nperseg=256, noverlap=128)
    audio_recovered = np.asarray(audio_recovered[:len(audio)])
    mse = np.append(mse, mean_squared_error(audio, audio_recovered))
    print(file, 'error=', mse[-1])
    sf.write(file+'_mag_only.wav', audio_recovered, _s)
print('average mse=', mse.sum() / len(mse))


# test for mag only, hann
print("mag only, hann")
mse = np.array([])
for file in clean_files:
    audio, _s = sf.read(file)
    audio = np.asarray(audio)
    _t, _f, Zxx = stft(audio, nperseg=256, noverlap=128, padded=True)
    _t, audio_recovered = istft(np.abs(Zxx), nperseg=256, noverlap=128)
    audio_recovered = np.asarray(audio_recovered[:len(audio)])
    mse = np.append(mse, mean_squared_error(audio, audio_recovered))
    print(file, 'error=', mse[-1])
    sf.write(file+'_mag_only_hann.wav', audio_recovered, _s)
print('average mse=', mse.sum() / len(mse))



# test for mag only, hann, using
# print("mag only, hann")
# mse = np.array([])
# for file in clean_files:
#     audio, _s = sf.read(file)
#     audio = np.asarray(audio)
#     _t, _f, Zxx = stft(audio, nperseg=256, noverlap=128, padded=True)
#     _t, audio_recovered = istft(np.abs(Zxx), nperseg=256, noverlap=128)
#     audio_recovered = np.asarray(audio_recovered[:len(audio)])
#     mse = np.append(mse, mean_squared_error(audio, audio_recovered))
#     print(file, 'error=', mse[-1])
#     sf.write(file+'_mag_only_hann.wav', audio_recovered, _s)
# print('average mse=', mse.sum() / len(mse))
