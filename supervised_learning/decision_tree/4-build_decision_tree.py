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
        """
        Initialise les noeuds avec les features optimal, threshold values,
        children, root status, and depth.
        """
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

    def count_nodes_below(self, only_leaves=False):

        """Compte les nœuds dans le sous-arbre à partir de ce nœud.
        Args:
        only_leaves (bool): Si True, ne compte que les feuilles.
                            Si False, compte aussi les noeuds internes.
        Returns:
        int: Nombre de nœuds ou de feuilles dans le sous-arbre.
        """
        l_count = self.left_child.count_nodes_below(only_leaves=only_leaves)
        r_count = self.right_child.count_nodes_below(only_leaves=only_leaves)

        if only_leaves:
            return l_count + r_count
        else:
            return 1 + l_count + r_count

    def update_bounds_below(self):
        """
        Met à jour les bornes (lower et upper) pour chaque
        nœud de l’arbre de décision.

        But :
        -------
        À chaque split, un nœud impose une contrainte
        supplémentaire sur la valeur
        possible de la feature utilisée pour la séparation.
        Les bornes représentent l’intervalle valide
        de valeurs pour chaque feature
        à l’intérieur de la région définie par ce nœud.

        Exemple :
        ---------
        Supposons un split sur feature 0 avec seuil = 30 :
            - Enfant gauche  : feature_0 ∈ [-∞ , 30]
            - Enfant droit   : feature_0 ∈ [30 , +∞]

        Si ensuite l’enfant droit split encore sur feature 0 avec seuil = 40 :
            - Enfant gauche  : feature_0 ∈ [30 , 40]
            - Enfant droit   : feature_0 ∈ [40 , +∞]

        Donc, plus on descend dans l’arbre, plus les intervalles se resserrent.

        Fonctionnement :
        ----------------
        1. Si le nœud est la racine :
            - lower = {feature : -∞}  (borne inférieure initiale)
            - upper = {feature : +∞}  (borne supérieure initiale)

        2. Chaque enfant hérite d’une copie des bornes du parent.

        3. Selon que l’enfant est gauche ou droit :
            - Enfant gauche → borne supérieure (upper) modifiée par le seuil
            - Enfant droit  → borne inférieure (lower) modifiée par le seuil

        4. Appel récursif pour propager cette mise às
        jour à tous les descendants.
        """
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1*np.inf}

        for child in [self.left_child, self.right_child]:

            child.lower = self.lower.copy()
            child.upper = self.upper.copy()

            if child is self.left_child:
                # borne supérieure modifiée
                child.lower[self.feature] = self.threshold
            else:
                # borne inférieure modifiée
                child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def __str__(self):
        """
        Returns a string representation of the node and it's children
        """
        node_type = "root" if self.is_root else "node"
        details = (f"{node_type} [feature={self.feature}, "
                   f"threshold={self.threshold}]\n")
        if self.left_child:
            left_str = self.left_child.__str__().replace("\n", "\n    |  ")
            details += f"    +---> {left_str}"

        if self.right_child:
            right_str = self.right_child.__str__().replace("\n", "\n       ")
            details += f"\n    +---> {right_str}"

        return details

    def get_leaves_below(self):
        """
        Fonction récursive qui pour trouvers les feuilles de
        chaque sous-abre et vas concaténer les 2 listes de feuilles
        """
        leaves_left = self.left_child.get_leaves_below()
        leaves_right = self.right_child.get_leaves_below()
        return leaves_left + leaves_right


class Leaf(Node):
    """Représente une feuille de l’arbre de décision.
    Une feuille ne possède pas d’enfants, seulement une valeur prédite
    et sa profondeur dans l’arbre.
    """
    def __init__(self, value, depth=None):
        """
        Initialise la feuille avec une valeur et profondeur spécifique
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        retourne la profondeur d'une feuille (self.depth)
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Retourne 1 car une feuille est toujours un nœud unique.
        Args:
        only_leaves (bool): Ignoré pour une feuille, toujours 1.
        Returns:
        int: 1
        """
        return 1

    def __str__(self, prefix="", is_left=True):
        """
        fonction qui print le résultat
        """
        return prefix + f"[value={self.value}]"

    def get_leaves_below(self):
        """
        Fonction qui retourne la feuille elle même
        """
        return [self]

    def update_bounds_below(self):
        pass


class Decision_Tree():
    """Classe principale représentant un arbre de décision.
    Attributs :
        max_depth (int): Profondeur maximale autorisée.
        min_pop (int): Taille minimale de population dans un nœud.
        rng (np.random.Generator): Générateur aléatoire pour les splits.
        split_criterion (str): Méthode de séparation (par défaut "random").
        root (Node): Racine de l’arbre.
    """
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """
        Initialise les paramètres de l'arbre décision
        """
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
        """Retourne la profondeur maximale de l’arbre.
        Cette méthode délègue le calcul à la racine de l’arbre.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Retourne le nombre de nœuds ou de feuilles dans l'arbre.
        Args:
            only_leaves (bool): Si True, ne compte que les feuilles.
        Returns:
            int: Nombre de nœuds ou de feuilles.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self,):
        return self.root.__str__() + "\n"

    def get_leaves(self):
        """
        appelle la fonction get_leaves_below() pour obtenir
        toutes feuilles à partir de root
        """
        return self.root.get_leaves_below()

    def update_bounds(self):
        self.root.update_bounds_below()
