#!/usr/bin/env python3
"""
Fonction qui retourne un nuage graphique
"""
import numpy as np
import matplotlib.pyplot as plt


def scatter():
    """
    Graphique de nuage pour une moyenne de poids
    """
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180
    plt.figure(figsize=(6.4, 4.8))

    # Spécification du graphique (couleur, titre)
    plt.scatter(x, y, color='magenta')
    plt.title("Men's Height vs Weight")
    plt.xlabel("Height (in)")
    plt.ylabel("Weight (lbs)")
    plt.savefig("1-scatter.png")
    plt.show()
