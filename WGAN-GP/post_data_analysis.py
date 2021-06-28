from numpy import genfromtxt
import matplotlib.pyplot as plt

if __name__ == "__main__":
 data_gen = genfromtxt("model/gen_loss.txt", delimiter=",")
 data_disc = genfromtxt("model/disc_loss.txt", delimiter=",")
 plt.plot(data_gen)
 plt.plot(data_disc)
 plt.legend(["generator loss", "discriminator loss"], loc='lower right')
 plt.show()

