#!/usr/bin/env python3
"""
Boucle interactive de questions-réponses qui se termine lorsque l’utilisateur saisit un mot-clé de sortie.
"""


def qa_loop():
    """
    Démarre une boucle interactive de questions-réponses.
    """
    exit_keywords = ['exit', 'quit', 'goodbye', 'bye']

    while True:
        # Obtenir l’entrée de l’utilisateur
        user_input = input("Q: ").strip()

        # Vérifier si l’entrée correspond à un mot-clé de sortie
        if user_input.lower() in exit_keywords:
            print("A: Goodbye")
            break

        # Réponse fictive pour l’exemple
        # Vous pouvez remplacer cette partie par une logique réelle pour générer des réponses
        print("A:")


if __name__ == "__main__":
    qa_loop()
