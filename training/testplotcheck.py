from AutoEncoder import AutoEncoder
from plotCheck import plot
import numpy	
#load model here
ae = AutoEncoder()
ae.load("AutoEncoder0")



with open("errort50l.txt","r") as f:
    for i in f:
        l = [float(p) for p in i.strip().split()]
        n = numpy.array([l])
        plot(n)
        plot(ae.predict(n))
        
