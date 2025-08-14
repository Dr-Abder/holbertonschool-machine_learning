#!/usr/bin/env python3
"""
Fonction qui trace une courbe y = x³ graphique linéaire
"""
import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    Trace la courbe de y en fonction de x
    """

    y = np.arange(0, 11) ** 3
    # Crée un tableau de 0 à 10 au cube (valeur de l'axe y)

    plt.figure(figsize=(6.4, 4.8))
    # Crée la nouvelle figure (zone graphique)

    plt.xlim(0, 10)
    # Crée un tableau de 0 à 10 (valeur de l'axe x)

    plt.plot(y, 'r-')
    # Trace les points x[i], y[i] (Chaque valeur x[i] correspond à y[i])
    # 'r-' = ligne rouge continue

    plt.savefig("0-line.png")
    plt.show()
    # Enregistre le graphique et le montre
