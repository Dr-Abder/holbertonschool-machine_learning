#!/usr/bin/env python3
"""
Arbre de décision
Composé de classe d'un arbre de décision
et de feuille
"""
import numpy as np


class Node:
    """Représente un nœud interne de l'arbre de décision.

    Un nœud contient une feature (attribut), un seuil de décision (threshold),
    et deux enfants (gauche et droit). Chaque nœud garde aussi en mémoire
    sa profondeur dans l’arbre et peut calculer la profondeur maximale
    des sous-arbres en dessous de lui.
    """

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):

        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """
        max_depth_below sert à trouver la profondeur maximale de l’arbre.
        Elle fonctionne grâce à la récursivité :
        Si le nœud est une feuille (Leaf), la profondeur est
        simplement self.depth.
        Si le nœud est un nœud interne (Node), on appelle la même méthode
        sur ses enfants gauche et droit.
        On compare les deux profondeurs et on renvoie le maximum.
        Ainsi, en partant de la racine (root), l’appel descend jusqu’aux
        feuilles puis remonte en prenant la profondeur la plus grande.
        """

        max_depth_left = self.left_child.max_depth_below()
        # Profondeur max du noeud de gauche
        max_depth_right = self.right_child.max_depth_below()
        # Profondeur max du noeud de droite
        return max(max_depth_left, max_depth_right)


class Leaf(Node):
    """Représente une feuille de l’arbre de décision.
    Une feuille ne possède pas d’enfants, seulement une valeur prédite
    et sa profondeur dans l’arbre.
    """
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        return self.depth   # retourne la profondeur d'une feuille (self.depth)


class Decision_Tree():
    """Classe principale représentant un arbre de décision.

    Attributs :
        max_depth (int): Profondeur maximale autorisée.
        min_pop (int): Taille minimale de population dans un nœud.
        rng (np.random.Generator): Générateur aléatoire pour les splits.
        split_criterion (str): Méthode de séparation (par défaut "random").
        root (Node): Racine de l’arbre.
    """
    def __init__(self, max_depth=10, min_pop=1,
                 seed=0, split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        return self.root.max_depth_below()
        """Retourne la profondeur maximale de l’arbre.
        Cette méthode délègue le calcul à la racine de l’arbre.
        """
