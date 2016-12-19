import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    
    return 1.0 / (1.0 + np.exp(-z))

def main():
    
    z = np.arange(-5, 5, 0.1)
    
    phi_z = sigmoid(z)
    
    plt.figure()
    plt.plot(z, phi_z)
    plt.axvline(x=0.0, color="k")
    plt.axhspan(0.0, 1.0, facecolor="1.0", alpha=0.5, ls="dotted")
    plt.axhline(y=0.5, ls="dotted", color="k")
    plt.yticks([0.0, 0.5, 1.0])
    plt.ylim(-0.1, 1.1)
    plt.xlabel("z")
    plt.ylabel("$\phi (z)$")
    plt.show()

if __name__ == "__main__":
    
    main()
