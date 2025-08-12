#!/usr/bin/env python3

"""
Fonction sert à faire un produit matriciel
"""

def mat_mul(mat1, mat2):


    if len(mat1[0]) == len(mat2):
        """
        Vérification si le nombre de colonne de mat1 est égal au 
        nombre de ligne de mat2 (mat1(m x n) et (n x p) = m x p,
        si n(mat1) et n(mat2) sont égaux)
        """

        m = len(mat1)
        n = len(mat2[0])
        """
        On attribue le nombre de ligne de mat1 à m
        On attribue le nombre de colonne à partir de la 
        ligne 1 (mat2[0]) de la matrice mat2 à n
        """

        mat3 = [[0 for _ in range(n)] for _ in range(m)]
        """
        Création de la matrice de mat3 de taille m(mat1) x p(mat2)
        et on mets 0 pour chaque valeur
        """

        for i in range(m):
            for j in range(n):
                for k in range(len(mat2)):
                    mat3[i][j] += mat1[i][k] * mat2[k][j]
        """
        On utilise trois boucles imbriquées pour remplir la matrice résultat.
        La première boucle avec i parcourt les lignes du résultat (et donc de mat1).
        Pour chaque ligne, la deuxième boucle avec j parcourt les colonnes du résultat (et donc de mat2).
        Enfin, pour chaque position (i, j), la troisième boucle avec k parcourt tous les 
        éléments nécessaires pour calculer la case : on prend les éléments de la ligne i dans mat1 
        et ceux de la colonne j dans mat2, on multiplie chaque paire correspondante et on additionne le tout.
        Ainsi, i choisit la ligne, j choisit la colonne, et k fait le “travail” du calcul.
        i → ligne du résultat
        j → colonne du résultat
        k → combine ligne i de mat1 avec colonne j de mat2
        """

        return mat3
    else:
        return None
