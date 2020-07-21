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
    padded = np.zeros((1306, 129, 1))
    padded[:1306, :prediction.shape[1], :prediction.shape[2]] = prediction[:1306, :prediction.shape[1], :prediction.shape[2]]
    squeeze = np.squeeze(padded)
    swap = np.swapaxes(squeeze, 0, 1)
    f = np.asarray([0.        , 0.00390625, 0.0078125 , 0.01171875, 0.015625  ,
       0.01953125, 0.0234375 , 0.02734375, 0.03125   , 0.03515625,
       0.0390625 , 0.04296875, 0.046875  , 0.05078125, 0.0546875 ,
       0.05859375, 0.0625    , 0.06640625, 0.0703125 , 0.07421875,
       0.078125  , 0.08203125, 0.0859375 , 0.08984375, 0.09375   ,
       0.09765625, 0.1015625 , 0.10546875, 0.109375  , 0.11328125,
       0.1171875 , 0.12109375, 0.125     , 0.12890625, 0.1328125 ,
       0.13671875, 0.140625  , 0.14453125, 0.1484375 , 0.15234375,
       0.15625   , 0.16015625, 0.1640625 , 0.16796875, 0.171875  ,
       0.17578125, 0.1796875 , 0.18359375, 0.1875    , 0.19140625,
       0.1953125 , 0.19921875, 0.203125  , 0.20703125, 0.2109375 ,
       0.21484375, 0.21875   , 0.22265625, 0.2265625 , 0.23046875,
       0.234375  , 0.23828125, 0.2421875 , 0.24609375, 0.25      ,
       0.25390625, 0.2578125 , 0.26171875, 0.265625  , 0.26953125,
       0.2734375 , 0.27734375, 0.28125   , 0.28515625, 0.2890625 ,
       0.29296875, 0.296875  , 0.30078125, 0.3046875 , 0.30859375,
       0.3125    , 0.31640625, 0.3203125 , 0.32421875, 0.328125  ,
       0.33203125, 0.3359375 , 0.33984375, 0.34375   , 0.34765625,
       0.3515625 , 0.35546875, 0.359375  , 0.36328125, 0.3671875 ,
       0.37109375, 0.375     , 0.37890625, 0.3828125 , 0.38671875,
       0.390625  , 0.39453125, 0.3984375 , 0.40234375, 0.40625   ,
       0.41015625, 0.4140625 , 0.41796875, 0.421875  , 0.42578125,
       0.4296875 , 0.43359375, 0.4375    , 0.44140625, 0.4453125 ,
       0.44921875, 0.453125  , 0.45703125, 0.4609375 , 0.46484375,
       0.46875   , 0.47265625, 0.4765625 , 0.48046875, 0.484375  ,
       0.48828125, 0.4921875 , 0.49609375, 0.5       ])

    t, pre = istft(swap, window='hamming', nperseg=256, noverlap=128)
    sf.write('test-1.wav', pre, 16000)
