from keras.layers import Input, Dense
from keras.models import Model , load_model
from keras.utils import plot_model
from keras.datasets import mnist
import numpy as np
import preprocess



class AutoEncoder(object):
    def __init__(self,inputSize,hiddenNumber):
        self.encoding_dim = hiddenNumber
        self.input_img = Input(shape=(inputSize,))
        self.encoded = Dense(self.encoding_dim,activation = 'relu')(self.input_img)

        self.outputLayer = Dense(inputSize,activation='sigmoid')(self.encoded)
        self.autoencoder = Model(self.input_img, self.outputLayer)

        self.encoder = Model(self.input_img, self.encoded)

        # create a placeholder for an encoded (32-dimensional) input
        encoded_input = Input(shape=(self.encoding_dim,))
        self.decoder_layer = self.autoencoder.layers[-1]
        self.decoder = Model(encoded_input, self.decoder_layer(encoded_input))
        self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


    def train(self,x_train,y_train,epochs):
        self.autoencoder.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size=256,
                        shuffle=True,
                        validation_data=(x_test, x_test))
        

    def predict(self,x_test):
        return self.autoencoder.predict(x_test)

    def save(self,name):
        self.autoencoder.save("../Model/"+name+".h5")

class StackedAutoEncoder(object):
    def __init__(self,inputSize,hiddenNumber,stackPercentages):
        self.aeL = [AutoEncoder(inputSize,hiddenNumber) for i in stackPercentages]
        input_img = Input(shape=(inputSize,))
        output = self.aeL[0].autoencoder(input_img)
        for i in range(1,len(self.aeL)):
            output = self.aeL[i].autoencoder(output)

        self.sae = Model(input_img,output)
        self.sae.compile(optimizer='adadelta', loss='binary_crossentropy')
        plot_model(self.sae, to_file='model.png')

    def train(self,x_train,y_train,epochs,percentagesList):
    	
        

    def predict(self,x_test):
        return self.sae.predict(x_test)

    def save(self,name):
        self.sae.save("../Model/"+name+".h5")

    def load(self,name):
        self.sae.load("../Model/"+name+".h5")


from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

ae = AutoEncoder(784,4000)
ae.train(x_train,x_train,10)

encoded_imgs = ae.predict(x_test)

import matplotlib.pyplot as plt
n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(encoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
