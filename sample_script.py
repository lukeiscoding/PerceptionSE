from models import *
import numpy as np
from glob import glob
import soundfile as sf
from scipy.signal import istft
if __name__ == '__main__':
    model_name = TFU_model_2
    model = model_name.model()
    model.summary()
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    file_list = sorted(glob("data/LibriSpeechProcessed/*.npy"))
    dev_ssn_3, dev_clean,test_ssn_3, test_clean = file_list
    # dev_ssn_3 = np.load(dev_ssn_3, allow_pickle=True)
    # dev_clean = np.load(dev_clean, allow_pickle=True)
    test_ssn_3 = np.load(test_ssn_3, allow_pickle=True)
    test_clean = np.load(test_clean, allow_pickle=True)
    model.load_weights('./checkpoints/my_checkpoint')

    prediction = model.predict(test_ssn_3[0])
    prediction = prediction.reshape((-1, 128, 1))
    padded = np.zeros((prediction.shape[0], 129, 1))
    padded[:, :128, 1] = prediction
    squeeze = np.squeeze(padded)
    swap = np.swapaxes(squeeze, 0, 1)
    t, prediction = istft(swap)
    print(prediction)
