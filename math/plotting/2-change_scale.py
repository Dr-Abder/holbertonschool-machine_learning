#!/usr/bin/env python3
"""
Fonction qui retourne un graphqie linéaire
avec un axe y en échelle logarithmique
"""
import numpy as np
import matplotlib.pyplot as plt


def change_scale():
    """
    Graphique linéaire de décroissance
    radioactive au bout de x année
    """
    x = np.arange(0, 28651, 5730)
    r = np.log(0.5)
    t = 5730
    y = np.exp((r / t) * x)
    plt.figure(figsize=(6.4, 4.8))

    # Spécification du graphique (couleur, titre, échelle, limite)
    plt.plot(x, y, 'b-')
    plt.yscale('log')
    plt.xlim(0, 28650)
    plt.title("Exponential Decay of C-14")
    plt.xlabel("Time (years)")
    plt.ylabel("Fraction Remaining")
    plt.savefig("2-change_scale.png")
    plt.show()
