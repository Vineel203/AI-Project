from keras.layers import Input, Dense
from keras.models import Model , load_model
from keras.utils import plot_model
from keras.datasets import mnist
import numpy as np

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
    def __init__(self,inputSize = 784,hiddenNumber = 2000,stackPercentages=["50","37","25","12","0"]):
        self.stackPercentages = stackPercentages
        self.aeL = [AutoEncoder(inputSize,hiddenNumber) for i in stackPercentages]

    def trainAutoEncoderl(self,no,epochs=10):
        x_train = []
        y_train = []
        x_file = "error"+self.stackPercentages[no]+"l.txt"
        y_file = "error"+self.stackPercentages[no+1]+"l.txt"
        print("opening "+x_file)
        with open(x_file,"r") as f:
            for l in f:
                x_train+=[[float(p) for p in l.strip().split()]]

        with open(y_file,"r") as f:
            for l in f:
                y_train+=[[float(p) for p in l.strip().split()]]

        tempCount = no+1
        while(tempCount<len(self.stackPercentages)):
            x_file = "error"+self.stackPercentages[tempCount]+"l.txt"
            print(x_file)
            with open(x_file,"r") as f:
                for l in f:
                    x_train+=[[float(p) for p in l.strip().split()]]
                    y_train+=[[float(p) for p in l.strip().split()]]
            tempCount+=1


        self.aeL[no].train(x_train,y_train,epochs)
        self.aeL[no].save("AutoEncoder"+str(no))        

    def predict(self,x_test):
        return self.sae.predict(x_test)

    def save(self,name):
        self.sae.save("../Model/"+name+".h5")

    def load(self,name):
        self.sae.load("../Model/"+name+".h5")

sa = StackedAutoEncoder()
sa.trainAutoEncoderl(3)