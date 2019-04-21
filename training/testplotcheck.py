from AutoEncoder import AutoEncoder , StackedAutoEncoder
from plotCheck import plot
import numpy	
#load model here
sae = StackedAutoEncoder()
sae.createStackAutoEncoder()
sae.load("StackedAutoEncoder")



with open("error50l.txt","r") as f:
    for i in f:
        l = [float(p) for p in i.strip().split()]
        n = numpy.array([l])
        plot(n)
        plot(sae.predict(n))
        
        
