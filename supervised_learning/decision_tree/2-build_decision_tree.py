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

    def left_child_add_prefix(self, text):
        lines = text.split("\n")           # on coupe le texte de l'enfant ligne par ligne
        new_text = "    +--" + lines[0] + "\n"   # première ligne avec le préfixe +--
        for x in lines[1:]:
            new_text += "   |  " + x + "\n"     # les autres lignes avec un petit | pour montrer la branche
        return new_text

    def right_child_add_prefix(self, text):
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "   " + x +"\n"
        return new_text

    def __str__(self):
        if self.is_root:
            node_text = f"root [feature={self.feature}, threshold={self.threshold}]"
        else :
            node_text = f"node [feature={self.feature}, threshold={self.threshold}]"
        node_text += self.left_child_add_prefix(self.left_child.__str__())
        node_text += self.right_child_add_prefix(self.right_child.__str__())
        return node_text

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

    def __str__(self):
        return (f"-> leaf [value={self.value}]")


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
        return self.root.__str__()
