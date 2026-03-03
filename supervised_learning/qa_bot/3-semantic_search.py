#!/usr/bin/env python3
"""
Recherche sémantique utilisant Sentence-BERT et la similarité cosinus
"""


import os
from sentence_transformers import SentenceTransformer
import numpy as np


def cosine_similarity(vec1, vec2):
    """
    Calcule la similarité cosinus entre deux vecteurs.
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def semantic_search(corpus_path, sentence):
    """
    Effectue une recherche sémantique sur un corpus de documents.
    """
    # Charger un modèle Sentence-BERT préentraîné
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Étape 1 : Lire les documents du corpus
    documents = []
    file_names = os.listdir(corpus_path)
    for file_name in file_names:
        if file_name.endswith('.md'):
            file_path = os.path.join(corpus_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                documents.append(f.read())

    # Étape 2 : Générer les embeddings pour les documents du corpus
    doc_embeddings = model.encode(documents)

    # Étape 3 : Générer un embedding pour la phrase d’entrée
    query_embedding = model.encode([sentence])[0]

    # Étape 4 : Calculer les similarités cosinus entre la requête et chaque document
    similarities = [cosine_similarity(query_embedding, doc_embedding)
                    for doc_embedding in doc_embeddings]

    # Étape 5 : Trouver l’indice du document ayant le score de similarité le plus élevé
    best_doc_index = np.argmax(similarities)

    # Retourner le document le plus similaire
    return documents[best_doc_index]
