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

    def update_indicator(self):
        """
        But :
            Mettre à jour l'attribut `self.indicator` qui est une fonction
            renvoyant un tableau booléen indiquant pour chaque individu
            s'il appartient à la région définie par ce nœud (intersection
            des contraintes `lower` et `upper`).

        Contexte :
            - Dans un arbre de décision, chaque feuille correspond à une
            sous-région de l’espace des features.
            - Les nœuds internes imposent des contraintes sur les features :
                * lower[k] : valeur minimale (exclusif) pour la feature k
                * upper[k] : valeur maximale (inclusif) pour la feature k
            - L'indicator est la fonction caractéristique de la région
            hyper-rectangulaire définie par ces bornes.

        Fonctionnement :
            1. is_large_enough(x) :
                - Retourne True pour un individu si toutes les features
                testées sont supérieures à leur borne inférieure.
                - Si aucune borne inférieure, retourne True pour tous.

            2. is_small_enough(x) :
                - Retourne True pour un individu si toutes les features
                testées sont inférieures ou égales à leur borne supérieure.
                - Si aucune borne supérieure, retourne True pour tous.

            3. Combinaison :
                - self.indicator = lambda x: np.all([is_large_enough(x),
                                                    is_small_enough(x)],
                                                    axis=0)
                - Renvoie True seulement si l'individu satisfait toutes les
                contraintes de lower ET upper.

        Exemple (2 features, 4 individus) :

            A = np.array([[1,22000],
                        [1,44000],
                        [0,22000],
                        [0,44000]])
            lower = {0: 0.5}       # feature0 > 0.5
            upper = {1: 30000}     # feature1 <= 30000

            Résultat attendu :
            Leaf0 (feature0 <= 0.5)          → [True, True, False, False]
            Leaf1 (feature0 > 0.5, f1 <= 30000) → [False, False, True, False]
            Leaf2 (feature0 > 0.5, f1 > 30000) → [False, False, False, True]
        """
        def is_large_enough(x):
            if not self.lower:
                return np.ones(x.shape[0], dtype=bool)

            conds = [x[:, k] <= self.upper[k] for k in self.upper.keys()]

            return np.all(np.array(conds), axis=0)

        def is_small_enough(x):
            if not self.upper:
                return np.ones(x.shape[0], dtype=bool)

            conds = [x[:, k] > self.lower[k] for k in self.lower.keys()]

            return np.all(np.array(conds), axis=0)

        self.indicator = lambda x: np.all(np.array([is_large_enough(x),
                                          is_small_enough(x)]), axis=0)

    def get_leaves_below(self):
        """
        Fonction récursive qui pour trouvers les feuilles de
        chaque sous-abre et vas concaténer les 2 listes de feuilles
        """
        leaves_left = self.left_child.get_leaves_below()
        leaves_right = self.right_child.get_leaves_below()
        return leaves_left + leaves_right

    def pred(self, x):
        """
        Sert vraiment à descendre récursivement dans l’arbre jusqu’à
        tomber sur une feuille qui contient la valeur de prédiction finale.
        """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


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
        """
        la fonction update_bounds_below s'applique que sur
        les noeuds et non une feuille car elle n’a pas de
        descendants à qui transmettre.
        """
        pass

    def pred(self, x):
        """
        Retourne le résultat de la prédiction
        """
        return self.value


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
        """
        appelle la fonction update_bounds_below() pour mettre
        à jour les bornes lower et upper
        """
        self.root.update_bounds_below()

    def update_predict(self):
        """
        Construit et met à jour la fonction de prédiction
        vectorisée de l’arbre.

        Étapes :
        1. Met à jour les bornes (lower/upper) de toutes les feuilles
           avec `update_bounds()`.
        2. Récupère toutes les feuilles de l’arbre.
        3. Pour chaque feuille, construit un indicateur booléen
           (`leaf.indicator`) qui permet de savoir si un individu
           appartient à la région de cette feuille.
        4. Définit une fonction `predict_func(A)` qui :
            - crée un tableau vide de prédictions
            - pour chaque feuille, assigne `leaf.value` aux individus
              détectés par `leaf.indicator(A)`
            - renvoie le tableau final des prédictions
        5. Associe cette fonction à `self.predict`.

        Avantages :
        - Contrairement à `pred(x)` qui prédit un seul individu
          de manière récursive, `predict(A)` est vectorisée et
          donc beaucoup plus efficace pour traiter de grandes matrices.
        """
        self.update_bounds()
        leaves = self.get_leaves()

        for leaf in leaves:
            leaf.update_indicator()

        def predict_func(A):
            predictions = np.zeros(A.shape[0], dtype=int)
            for leaf in leaves:
                predictions[leaf.indicator(A)] = leaf.value
            return predictions

        self.predict = predict_func

    def pred(self, x):
        """
        Retourne le résultat de la prédiction
        """
        return self.root.pred(x)

    def fit(self, explanatory, target, verbose=0):
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion
        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)

        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}
    - Accuracy on training data : {self.accuracy(self.explanatory,
                                   self.target)}""")

    def np_extrema(self, arr):
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            sub_pop = self.explanatory[:, feature][node.sub_population]
            feature_min, feature_max = self.np_extrema(sub_pop)
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def fit_node(self, node):
        node.feature, node.threshold = self.split_criterion(node)

        left_population = (
            node.sub_population &
            (self.explanatory[:, node.feature] > node.threshold)
        )
        right_population = (
            node.sub_population &
            (self.explanatory[:, node.feature] <= node.threshold)
        )

        # Is left node a leaf ?
        is_left_leaf = (
            (np.sum(left_population) < self.min_pop) or
            (node.depth + 1 >= self.max_depth) or
            (len(np.unique(self.target[left_population])) == 1)
        )

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Is right node a leaf ?
        is_right_leaf = (
            (np.sum(right_population) < self.min_pop) or
            (node.depth + 1 >= self.max_depth) or
            (len(np.unique(self.target[right_population])) == 1)
        )
        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        if sub_population.sum() == 0:
            values = self.target[node.sub_population]
        else:
            values = self.target[sub_population]

        value = np.bincount(values).argmax()

        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        return (
            np.sum(np.equal(self.predict(test_explanatory), test_target)) /
            test_target.size
        )
