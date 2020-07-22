from scipy.signal import stft, istft
from glob import glob
import soundfile as sf
import numpy as np
from sklearn.metrics import mean_squared_error

clean_files = sorted(glob('stft_test_file/*.flac'))

# test for stft and istft
print("full complex number")
mse = np.array([])
for file in clean_files:
    audio, _s = sf.read(file)
    audio = np.asarray(audio)
    # print("audio length=", len(audio)," ", len(audio)/16000, "s")
    _t, _f, Zxx = stft(audio, window='hamming', nperseg=256, noverlap=128, padded=True)
    # print("shape after stft =", Zxx.shape,"total: ",Zxx.shape[0],"*",Zxx.shape[1],"=",Zxx.shape[0]*Zxx.shape[1])
    _t, audio_recovered = istft(Zxx, window='hamming', nperseg=256, noverlap=128)
    audio_recovered = np.asarray(audio_recovered[:len(audio)])
    mse = np.append(mse, mean_squared_error(audio, audio_recovered))
    print(file, "lossless", 'error=', mse[-1])

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
    # print(file, 'error=', mse[-1])
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
    # print(file, 'error=', mse[-1])
    sf.write(file+'_mag_only_hann.wav', audio_recovered, _s)
print('average mse=', mse.sum() / len(mse))



# test for mag only, hamming, using griffin-lim
import librosa
import matplotlib.pyplot as plt
print("mag only, hamming, using griffin-lim")
mse = np.array([])
for file in clean_files:
    audio, _s = sf.read(file)
    audio = np.asarray(audio)
    _t, _f, Zxx = stft(audio, nperseg=256, noverlap=128, padded=False)
    Zxx2 = librosa.stft(audio, window='hann', n_fft=256, win_length=256, hop_length=128)
    plt.figure()
    ax = plt.subplot(4,1,1)
    plt.imshow(np.abs(Zxx), aspect="auto")
    plt.title('scipy stft spectrogram')
    plt.subplot(4,1,2, sharex=ax, sharey=ax)
    plt.imshow(np.abs(Zxx2), aspect="auto")
    plt.title('librosa stft spectrogram')
    Mag = np.abs(Zxx2)
    audio_recovered = librosa.core.griffinlim(Mag, n_iter=1000, momentum=0, window="hann", win_length=256, hop_length=128, length=audio.shape[0])
    print(audio.shape, Zxx.shape, audio_recovered.shape)
    # audio_recovered = np.asarray(audio_recovered[:len(audio)])
    mse = np.append(mse, mean_squared_error(audio, audio_recovered))
    print(file, 'error=', mse[-1])
    sf.write(file+'_mag_only_hamming_griffin.wav', audio_recovered, _s)
    ax = plt.subplot(4,1,3)
    plt.plot(audio)
    plt.title('Original')
    plt.xlabel('')
    plt.subplot(4,1,4, sharex=ax, sharey=ax)
    plt.plot(audio_recovered)
    plt.title('Griffin-Lim reconstruction')
    plt.xlabel('')
    plt.tight_layout()
    plt.show()
print('average mse=', mse.sum() / len(mse))
