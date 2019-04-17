from AutoEncoder import AutoEncoder
from plotCheck import plot
import numpy	
#load model here
ae = AutoEncoder()
ae.load("AutoEncoder1")



with open("error37l.txt","r") as f:
    for i in f:
        l = [float(p) for p in i.strip().split()]
        n = numpy.array([l])
        plot(ae.predict(n))
        break
