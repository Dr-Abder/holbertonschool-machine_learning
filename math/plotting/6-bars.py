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
    fruit = np.random.randint(0, 20, (4,3))
    plt.figure(figsize=(6.4, 4.8))

    # Spécification du graphique (couleur, titre, échelle, limite)
    personne = ['Farrah', 'Fred', 'Felicia']
    couleur = ['red', 'yellow', '#ff8000','#ffe5b4']
    fruit_labels = ['apples', 'bananas', 'oranges', 'peaches']

    barres = np.arange(3)

    # Spécification du graphique (couleur, titre, échelle, limite)
    plt.bar(barres, fruit[0], color=couleur[0], width=0.5)
    plt.bar(barres, fruit[1], color=couleur[1], width=0.5, bottom=fruit[0])
    plt.bar(barres, fruit[2], color=couleur[2], width=0.5, bottom=fruit[0]+fruit[1])
    plt.bar(barres, fruit[3], color=couleur[3], width=0.5, bottom=fruit[0]+fruit[1]+fruit[2])
    plt.xticks(barres, personne)
    plt.ylabel("Quantity of Fruit")
    plt.yticks(np.arange(0, 81, 10))
    plt.title("Number of Fruit per Person")
    plt.legend(fruit_labels)

    # Show the plot
    plt.savefig("6-bars.png")
    plt.show()
