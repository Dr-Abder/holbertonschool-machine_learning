#!/usr/bin/env python3
"""
La régularisation L2 (également appelée régression de crête)
est une technique permettant de réduire la complexité
d'un modèle en diminuant les poids d'un modèle proportionnellement
au carré de chaque poids (pénalisant donc particulièrement ceux qui
sont les plus élevés), mais ne les rend pas nuls.
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calcule le coût régularisé avec la pénalisation L2.

    La régularisation L2 ajoute une pénalité proportionnelle à la somme
    des carrés des poids du réseau de neurones, afin de réduire le risque
    de surapprentissage.
        Formule mathématique :

    Formule mathématique :
        J = cost + (λ / 2m) * Σ ||Wᵢ||²

    Args:
        cost (float): Coût initial (par exemple issu de l'entropie croisée).
        lambtha (float): Paramètre de régularisation L2.
        weights (dict): Dictionnaire contenant les matrices de poids de
            chaque couche, avec les clés au format 'W1', 'W2', ..., 'WL'.
        L (int): Nombre total de couches du réseau.
        m (int): Nombre d'exemples d'apprentissage.

    Returns:
        float: Nouveau coût après ajout du terme de régularisation L2.
    """
    W_somme = 0
    for i in range(1, L + 1):
        W_couche = weights[f'W{i}']
        W_somme += (W_couche ** 2).sum()

    regularization_ter = (lambtha / (2 * m)) * W_somme
    new_cost = cost + regularization_ter

    return new_cost
