#!/usr/bin/env python3
"""
Boucle de questions-réponses utilisant BERT préentraîné,
qui se termine sur des mots-clés spécifiques.
"""

# Importer la fonction 'question_answer' depuis '0-qa.py'
qa_module = __import__('0-qa')
question_answer = qa_module.question_answer  # Extraire la fonction


def answer_loop(reference):
    """
    Boucle interactive qui répond aux questions à partir d’un texte de référence.
    """
    exit_keywords = ['exit', 'quit', 'goodbye', 'bye']

    while True:
        # Obtenir l’entrée de l’utilisateur
        user_input = input("Q: ").strip()

        # Vérifier si l’entrée correspond à un mot-clé de sortie
        if user_input.lower() in exit_keywords:
            print("A: Goodbye")
            break

        # Appeler la fonction question_answer pour obtenir la réponse
        answer = question_answer(user_input, reference)

        # Si aucune réponse valide n’est trouvée, retourner une réponse par défaut
        if answer:
            print(f"A: {answer}")
        else:
            print("A: Sorry, I do not understand your question.")
