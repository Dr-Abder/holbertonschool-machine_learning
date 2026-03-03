#!/usr/bin/env python3
"""
Système de questions-réponses utilisant la recherche sémantique et BERT.
"""

import os
from sentence_transformers import SentenceTransformer
import numpy as np

qa_module = __import__('0-qa')
extract_answer = qa_module.question_answer


def cosine_similarity(vec1, vec2):
    """
    Calcule la similarité cosinus entre deux vecteurs.
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def load_corpus(corpus_path):
    """
    Charge les fichiers texte du répertoire corpus_path dans une liste de chaînes de caractères.
    """
    corpus = []
    for filename in os.listdir(corpus_path):
        file_path = os.path.join(corpus_path, filename)
        if file_path.endswith('.md'):  # Charger uniquement les fichiers markdown
            with open(file_path, 'r', encoding='utf-8') as f:
                corpus.append(f.read())

    return corpus


def semantic_search(corpus, sentence):
    """
    Effectue une recherche sémantique sur un corpus de documents.
    """
    # Charger un modèle Sentence-BERT préentraîné
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Générer les embeddings pour les documents du corpus
    doc_embeddings = model.encode(corpus)

    # Générer un embedding pour la phrase d’entrée
    query_embedding = model.encode([sentence])[0]

    # Calculer les similarités cosinus entre la requête et chaque document
    similarities = [cosine_similarity(query_embedding, doc_embedding)
                    for doc_embedding in doc_embeddings]

    # Trouver l’indice du document ayant le score de similarité le plus élevé
    best_doc_index = np.argmax(similarities)

    # Retourner le document le plus similaire
    return corpus[best_doc_index]

def question_answer(corpus_path):
    """
    Répond de manière interactive aux questions à partir de plusieurs textes de référence.
    """
    # Charger les documents du corpus
    corpus = load_corpus(corpus_path)

    # Mots-clés de sortie
    exit_keywords = ['exit', 'quit', 'goodbye', 'bye']

    while True:
        # Obtenir l’entrée de l’utilisateur
        user_input = input("Q: ").strip()

        # Vérifier si l’utilisateur souhaite quitter
        if user_input.lower() in exit_keywords:
            print("A: Goodbye")
            break

        # Effectuer une recherche sémantique pour trouver le document le plus pertinent
        relevant_doc = semantic_search(corpus, user_input)

        # Utiliser la fonction question_answer pour extraire la réponse du document pertinent
        answer = extract_answer(user_input, relevant_doc)

        # Si aucune réponse valide n’est trouvée, retourner une réponse par défaut
        if answer:
            print(f"A: {answer}")
        else:
            print("A: Sorry, I do not understand your question.")
