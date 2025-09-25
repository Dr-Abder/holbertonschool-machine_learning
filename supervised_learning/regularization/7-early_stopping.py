#!/usr/bin/env python3
"""
Ce module implémente la logique d’early stopping (arrêt anticipé)
pour l’entraînement d’un modèle. Cette technique permet de stopper
l’entraînement lorsque le coût ne s’améliore plus, afin de limiter
le surapprentissage et de gagner du temps de calcul.
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Détermine si l’entraînement doit être interrompu selon la stratégie
    d’early stopping.

    Args:
        cost (float): Coût actuel du modèle (ex. perte sur validation).
        opt_cost (float): Meilleur coût observé jusqu’à présent.
        threshold (float): Seuil minimal de différence pour considérer
                           une amélioration significative.
        patience (int): Nombre maximal d’itérations consécutives sans
                        amélioration avant d’arrêter l’entraînement.
        count (int): Compteur d’itérations consécutives sans amélioration.

    Returns:
        tuple:
            - stop (bool): True si l’entraînement doit être arrêté sinon False.
            - count (int): Nouveau compteur mis à jour après cette itération.

    Notes:
        - Si `cost` s’améliore d’au moins `threshold` par rapport à
          `opt_cost`, le compteur `count` est remis à zéro.
        - Sinon, `count` est incrémenté. Si `count >= patience`,
          l’entraînement est arrêté (stop=True).
    """
    if cost < opt_cost - threshold:
        count = 0
    else:
        count += 1

    if count >= patience:
        return True, count
    else:
        return False, count
