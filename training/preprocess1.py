from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.

count = 30000
ci = 0

#write 0 % error
with open("error0l.txt","w") as f:
	for matrix in x_train:
		for row in range(28):
			for colomn in range(28):
				f.write(str(matrix[row][colomn])+" ")
		f.write("\n")
		ci+=1
		if(ci>count):
			break


percentages = [50, 37, 25, 12]
perrows = [14,11,7,3]


#write l% error
for i in range(4):
	perr = perrows[i]
	ci = 0
	with open("error"+str(percentages[i])+"l.txt","w") as f:
		for matrix in x_train:
			for row in range(28):
				for colomn in range(28):
					if(colomn<perr):
						f.write(str(0)+" ")
					else:
						f.write(str(matrix[row][colomn])+" ")
			f.write("\n")
			ci+=1
			if(ci>count):
				break
