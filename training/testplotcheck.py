from AutoEncoder import AutoEncoder , StackedAutoEncoder
from plotCheck import plot
from keras.models import load_model
import numpy
#load model here
sae = StackedAutoEncoder(hiddenNumber = 100,optimizer ='adam',loss ='mse' )
sae.createStackAutoEncoder()
sae.load("Stacked_Auto_encoder")



with open("error50l.txt","r") as f:
    for i in f:
        l = [float(p) for p in i.strip().split()]
        n = numpy.array([l])
        plot(n)
        #plot(model.predict(n))
        #plot(n)
        n=sae.aeL[0].predict(n)
        plot(n)
        n=sae.aeL[1].predict(n)
        plot(n)
        n=sae.aeL[2].predict(n)
        plot(n)
        n=sae.aeL[3].predict(n)
        plot(n)
