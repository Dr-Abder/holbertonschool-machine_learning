#!/usr/bin/env python3
"""
Question-Réponse avec BERT préentraîné
"""

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """
    Trouve un extrait de texte dans un document de référence pour répondre à une question.
    """

    print("Initializing BERT Tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(
            "bert-large-uncased-whole-word-masking-finetuned-squad")

    print("Loading BERT model from TensorFlow Hub...")
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    # Tokeniser les entrées à l’aide du tokenizer BERT
    print("Tokenizing the question and reference document...")
    max_len = 512  # Longueur maximale de tokens pour BERT
    inputs = tokenizer(question, reference, return_tensors="tf")

    # Préparer les tenseurs d’entrée pour le modèle TensorFlow Hub
    input_tensors = [
            inputs["input_ids"],      # Identifiants des tokens
            inputs["attention_mask"], # Masque pour les tokens de remplissage (padding)
            inputs["token_type_ids"]  # Identifiants de type de token pour distinguer la question du contexte
            ]

    print("Running inference on the model...")
    # Passer les tenseurs d’entrée au modèle BERT QA et récupérer les logits de début et de fin
    output = model(input_tensors)

    # Accéder aux logits de début et de fin
    start_logits = output[0]
    end_logits = output[1]

    # Obtenir la longueur de la séquence d’entrée
    sequence_length = inputs["input_ids"].shape[1]
    print(f"Input sequence length: {sequence_length}")

    # Trouver les meilleurs indices de début et de fin dans la séquence d’entrée
    print("Determining the best start and end indices for the answer...")
    start_index = tf.math.argmax(start_logits[0, 1:sequence_length - 1]) + 1
    end_index = tf.math.argmax(end_logits[0, 1:sequence_length - 1]) + 1
    print(f"Start index: {start_index}, End index: {end_index}")

    # Récupérer les tokens de la réponse à l’aide des meilleurs indices
    print("Extracting the answer tokens...")
    answer_tokens = inputs["input_ids"][0][start_index: end_index + 1]

    # Décoder les tokens de la réponse pour obtenir la réponse finale
    print("Decoding the answer tokens...")
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True,
                              clean_up_tokenization_spaces=True)

    # Si aucune réponse n’est trouvée (c.-à-d. réponse vide ou composée d’espaces), retourner None
    if not answer.strip():
        print("No valid answer found.")
        return None

    print(f"Answer: {answer}")
    return answer
