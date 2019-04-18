import matplotlib.pyplot as plt
import numpy
def plot(encoded_imgs):
    plt.figure(figsize=(20, 4))
    plt.gray()
    plt.imshow(encoded_imgs.reshape(28, 28))
    plt.show()
