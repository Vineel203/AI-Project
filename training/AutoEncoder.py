from keras.layers import Input, Dense
from keras.models import Model , load_model
from keras.utils import plot_model
from keras.datasets import mnist
import numpy as np
from plotCheck import plot
import random

class AutoEncoder(object):
    def __init__(self,inputSize=784,hiddenNumber=2000):
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
        self.autoencoder.compile(optimizer='sgd', loss='mean_squared_error' , metrics=['mae', 'acc'])


    def train(self,x_train,y_train,epochs):
        self.autoencoder.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size=256,
                        shuffle=True
                        )
        

    def predict(self,x_test):
        return self.autoencoder.predict(x_test)

    def save(self,name):
        self.autoencoder.save("../Model/"+name+".h5")

    def load(self,name):
        self.autoencoder.load_weights("../Model/"+name+".h5")

class StackedAutoEncoder(object):
    def __init__(self,inputSize = 784,hiddenNumber = 2000,stackPercentages=["50","37","25","12","0"]):
        self.inputSize = inputSize
        self.stackPercentages = stackPercentages
        self.aeL = [AutoEncoder(inputSize,hiddenNumber) for i in stackPercentages]

    def trainAutoEncoderl(self,no,epochs=10):
        (xtrain, _), (_, _) = mnist.load_data()
        xtrain = xtrain.astype('float32') / 255.
        perList = [14,11,7,3,0]

        x_train = []
        y_train = []
        x_file = "error"+self.stackPercentages[no]+"l.txt"
        y_file = "error"+self.stackPercentages[no+1]+"l.txt"
        print("opening "+x_file)
        #loading no
        perr = perList[no]
        temp = []

        line = 0
        #for matrix in xtrain:
        while(True):
            ri = random.randint(0,59999)
            matrix = xtrain[ri]
            line+=1
            print(line)
            temp = []
            for row in range(28):
                for colomn in range(28):
                    if(colomn<perr):
                        temp.append(float(0))
                    else:
                        temp.append(matrix[row][colomn])
            x_train.append(temp)
            if(line>2000):
                break
        # loading no+1
        perr = perList[no+1]
        line = 0
        #for matrix in xtrain:
        while(True):
            ri = random.randint(0,59999)
            matrix = xtrain[ri]
            line+=1
            print(line)
            temp = []
            for row in range(28):
                for colomn in range(28):
                    if(colomn<perr):
                        temp.append(float(0))
                    else:
                        temp.append(matrix[row][colomn])
            y_train.append(temp)
            if(line>2000):
                break

        #loading sames
        tempCount = no+1
        while(tempCount<len(self.stackPercentages)):
            x_file = "error"+self.stackPercentages[tempCount]+"l.txt"
            perr = perList[tempCount]
            line = 0
            #for matrix in xtrain:
            while(True):
                ri = random.randint(0,59999)
                matrix = xtrain[ri]
                line+=1
                print(line)
                temp = []
                for row in range(28):
                    for colomn in range(28):
                        if(colomn<perr):
                            temp.append(float(0))
                        else:
                            temp.append(matrix[row][colomn])
                x_train.append(temp)
                y_train.append(temp)
                if(line>2000):
                    break
            tempCount+=1

        print("training started!")
        self.aeL[no].train(np.array(x_train),np.array(y_train),epochs)
        self.aeL[no].save("AutoEncoder"+str(no))
        
        

    def predict(self,x_test):
        return self.sae.predict(x_test)

    def save(self,name):
        self.sae.save("../Model/"+name+".h5")

    def load(self,name):
        self.sae.load_weights("../Model/"+name+".h5")

    def train(self,epochs):
        (xtrain, _), (_, _) = mnist.load_data()
        perList = [14,11,7,3,0]
        xtrain = xtrain.astype('float32') / 255.
        line =0
        x_train = []
        y_train = []
        for i in range(len(perList)):
            perr = perList[i]
            line = 0
            #for matrix in xtrain:
            while(True):
                ri = random.randint(0,59999)
                matrix = xtrain[ri]
                line+=1
                print(line)
                temp = []
                temp1 = []
                for row in range(28):
                    for colomn in range(28):
                        temp1.append(matrix[row][colomn])
                        if(colomn<perr):
                            temp.append(float(0))
                        else:
                            temp.append(matrix[row][colomn])
                x_train.append(temp)
                y_train.append(temp1)
                if(line>5000):
                    break
        print("training Started ..! ")
        self.sae.fit(np.array(x_train), np.array(y_train),
                        epochs=epochs,
                        batch_size=256,
                        shuffle=True
                        )

    def evaluate(self):
        (_, _), (xtrain, _) = mnist.load_data()
        perList = [14,11,7,3,0]
        xtrain = xtrain.astype('float32') / 255.
        line =0
        x_train = []
        y_train = []
        for i in range(len(perList)):
            perr = perList[i]
            line = 0
            for matrix in xtrain:
                line+=1
                print(line)
                temp = []
                temp1 = []
                for row in range(28):
                    for colomn in range(28):
                        temp1.append(matrix[row][colomn])
                        if(colomn<perr):
                            temp.append(float(0))
                        else:
                            temp.append(matrix[row][colomn])
                x_train.append(temp)
                y_train.append(temp1)
                if(line>5000):
                    break
        print("evaluate Started ..! ")
        return self.sae.evaluate(x_train, y_train)


    def createStackAutoEncoder(self):
        input_img = Input(shape=(self.inputSize,))
        output = self.aeL[0].autoencoder(input_img)
        output1 = self.aeL[1].autoencoder(output)
        output2 = self.aeL[2].autoencoder(output1)
        output3 = self.aeL[3].autoencoder(output2)

        self.sae = Model(input_img,output3)
        self.sae.compile(optimizer='sgd', loss='mean_squared_error' , metrics=['mae', 'acc'])

if __name__ == "__main__":
    sa = StackedAutoEncoder()
    sa.trainAutoEncoderl(0,30)
    sa.trainAutoEncoderl(1,30)
    sa.trainAutoEncoderl(2,30)
    sa.trainAutoEncoderl(3,30)
    sa.createStackAutoEncoder()
    sa.train(30)
    sa.save("StackedAutoEncoder")
    
