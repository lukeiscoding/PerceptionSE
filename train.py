from keras import *
from keras.layers import *

def TFU_model():
    image = Input(shape=(128, 128, 1))
    conv_1 = Conv2D(16, kernel_size=(7, 7), activation='relu', name='conv_1')(image)
    conv_2 = Conv2D(32, kernel_size=(5, 5), activation='relu', name='conv_2')(conv_1)
    conv_3 = Conv2D(64, kernel_size=(5, 5), activation='relu', name='conv_3')(conv_2)
    conv_4 = Conv2D(128, kernel_size=(3, 3), strides=2, activation='relu', name='conv_4')(conv_3)
    conv_5 = Conv2D(128, kernel_size=(3, 3), strides=2, activation='relu', name='conv_5')(conv_4)
    conv_6 = Conv2D(128, kernel_size=(3, 3), strides=2, activation='relu', name='conv_6')(conv_5)

    # doub_6 = Deconv2D(128, kernel_size=(2, 2), strides=2, activation='relu', name='doub_6')(conv_6)
    deconv_6 = Deconv2D(128, kernel_size=(3, 3), strides=2, activation='relu', name='deconv_6')(conv_6)
    deconv_6 = Concatenate()([conv_5, deconv_6])
    deconv_5 = Deconv2D(128, kernel_size=(4, 4), strides=2, activation='relu', name='deconv_5')(deconv_6)
    deconv_5 = Concatenate()([conv_4, deconv_5])
    deconv_4 = Deconv2D(64, kernel_size=(4, 4), strides=2, activation='relu', name='deconv_4')(deconv_5)
    deconv_4 = Concatenate()([conv_3, deconv_4])
    deconv_3 = Deconv2D(32, kernel_size=(5, 5), activation='relu', name='deconv_3')(deconv_4)
    deconv_3 = Concatenate()([conv_2, deconv_3])
    deconv_2 = Deconv2D(16, kernel_size=(5, 5), activation='relu', name='deconv_2')(deconv_3)
    deconv_2 = Concatenate()([conv_1, deconv_2])
    deconv_1 = Deconv2D(1, kernel_size=(7, 7), activation='relu', name='deconv_1')(deconv_2)
    deconv_1 = Concatenate()([image, deconv_1])
    deconv_0 = Deconv2D(1, kernel_size=(1, 1), activation='relu', name='deconv_0')(deconv_1)
    output = deconv_0
    model = Model(input=image, output=output)
    return model

if __name__ == '__main__':
    model = TFU_model()
    model.summary()
    print("hello world")