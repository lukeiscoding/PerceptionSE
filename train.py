from models import *

if __name__ == '__main__':
    model_name = TFU_model_2
    model = model_name.model()
    model.summary()
    print("hello world")