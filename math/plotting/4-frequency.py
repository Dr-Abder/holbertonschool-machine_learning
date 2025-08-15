#!/usr/bin/env python3
"""
graphique a deux courbe
"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    Fonction qui retoure deux courbe
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    # Spécification du graphique (couleur, titre, échelle, limite)
    bins = np.arange(0, 101, 10)
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')
    plt.xlim(0, 100)
    plt.ylim(0, 30)
    plt.hist(student_grades, bins, edgecolor='black')
    plt.xticks(np.arange(0, 110, 10))
    plt.savefig("4-frequency.png")
    plt.show()
