#!/usr/bin/env python3
"""
This function generates a stacked bar graph
"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    Requires matplotlib and numpy libraries
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    # Spécification du graphique (couleur, titre, échelle, limite)
    personne = ['Farrah', 'Fred', 'Felicia']
    couleur = ['red', 'yellow', '#ff8000', '#ffe5b4']
    fruit_labels = ['apples', 'bananas', 'oranges', 'peaches']

    barres = np.arange(3)

    # Create stacked bars
    for idx, row in enumerate(fruit):
        plt.bar(people, row, color=colors[idx], label=fruits[idx],
                bottom=bottom, width=0.5)
        bottom += row

    # Labeling and setting limits
    plt.ylabel('Quantity of Fruit')
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))
    plt.title('Number of Fruit per Person')

    # Show the plot
    plt.savefig("6-bars.png")
    plt.show()
