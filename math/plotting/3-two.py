#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
"""
Fonction qui retoure deux courbe
"""


def two():
    """
    graphique a deux courbe
    """
    x = np.arange(0, 21000, 1000)
    r = np.log(0.5)
    t1 = 5730
    t2 = 1600
    y1 = np.exp((r / t1) * x)
    y2 = np.exp((r / t2) * x)
    plt.figure(figsize=(6.4, 4.8))

    # Spécification du graphique (couleur, titre, échelle, limite)
    plt.plot(x, y1, 'r--', label='C-14')
    plt.plot(x, y2, 'g-', label='Ra-226')
    plt.title("Exponential Decay of C-14")
    plt.xlabel("Time (years)")
    plt.ylabel("Fraction Remaining")
    plt.xlim(0, 20000)
    plt.ylim(0, 1)
    plt.legend(loc="upper right")

    plt.savefig("3-two.png")
    plt.show()
