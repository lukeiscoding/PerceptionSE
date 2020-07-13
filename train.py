from models import *
import numpy as np

if __name__ == '__main__':
    model_name = TFU_model_2
    model = model_name.model()
    model.summary()
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    model.fit()