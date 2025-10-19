# Transfer Learning - CIFAR-10 Classification

## Description

Ce projet implémente un modèle de classification d'images utilisant le **transfer learning** avec MobileNetV2 sur le dataset CIFAR-10. Le modèle atteint une précision de validation de **90.79%**.

## Architecture du Modèle

- **Modèle de base** : MobileNetV2 pré-entraîné sur ImageNet
- **Taille des images** : Redimensionnement de 32×32 à 96×96
- **Data Augmentation** : Flip horizontal, rotation, zoom, contraste
- **Couches ajoutées** :
  - GlobalAveragePooling2D
  - Dense(256, activation='relu')
  - Dropout(0.3)
  - Dense(10, activation='softmax')

## Stratégie d'Entraînement

### Phase 1 : Entraînement initial (base gelée)
- Toutes les couches de MobileNetV2 sont gelées
- 15 epochs maximum avec EarlyStopping
- Batch size : 64
- Optimizer : Adam (learning rate par défaut)

### Phase 2 : Fine-tuning
- Dégelage des 30 dernières couches de MobileNetV2
- 15 epochs maximum avec EarlyStopping
- Batch size : 64
- Optimizer : Adam (learning rate = 1e-4)

## Résultats

- **Précision de validation finale** : **90.79%**
- **Objectif requis** : 87% ✅
- **Total epochs effectués** : ~30 (variable selon EarlyStopping)

## Prérequis

```bash
Python 3.9+
TensorFlow 2.15
NumPy 1.25.2
```

## Structure des Fichiers

```
transfer_learning/
├── 0-transfer.py    # Script d'entraînement du modèle
├── 0-main.py        # Script de test du modèle
├── cifar10.h5       # Modèle entraîné (généré après entraînement)
└── README.md        # Ce fichier
```

## Instructions d'Utilisation

### 1. Entraîner le Modèle

**Option A : Sur Google Colab (Recommandé - ~10-15 min avec GPU)**

1. Ouvrir [Google Colab](https://colab.research.google.com)
2. Activer le GPU : `Runtime → Change runtime type → GPU → T4`
3. Copier le contenu de `0-transfer.py` dans une cellule
4. Exécuter la cellule
5. Télécharger le fichier `cifar10.h5` généré :
   ```python
   from google.colab import files
   files.download('cifar10.h5')
   ```
6. Placer le fichier `cifar10.h5` dans le dossier du projet

**Option B : En local (CPU uniquement - ~1h30)**

```bash
./0-transfer.py
```

### 2. Tester le Modèle

Une fois le modèle entraîné et le fichier `cifar10.h5` présent :

```bash
./0-main.py
```

**Sortie attendue :**
```
79/79 ━━━━━━━━━━━━━━━━━━━━ 28s 325ms/step - accuracy: 0.9079 - loss: 0.3048
```

## Notes Techniques

### Adaptation du Script de Test

Le script `0-main.py` a été adapté pour résoudre des problèmes de compatibilité :

1. **`K.learning_phase` obsolète** : Ajout d'une condition pour les versions récentes de Keras
2. **Désérialisation de la couche Lambda** : 
   - Ajout de `safe_mode=False` dans `load_model()`
   - Import de `tensorflow` dans `builtins` pour rendre `tf` accessible globalement
3. **Compatibilité avec les fonctions lambda** : Nécessaire pour le redimensionnement des images

### Fonction `preprocess_data`

Cette fonction est exportée et utilisable dans d'autres scripts :

```python
from 0-transfer import preprocess_data

# Prétraite les images selon MobileNetV2
# Encode les labels en one-hot (10 classes)
X_preprocessed, Y_preprocessed = preprocess_data(X, Y)
```

## Optimisations Appliquées

- ✅ Data augmentation (flip, rotation, zoom, contraste)
- ✅ Dropout pour la régularisation
- ✅ EarlyStopping pour éviter l'overfitting
- ✅ ModelCheckpoint pour sauvegarder le meilleur modèle
- ✅ Fine-tuning progressif avec learning rate réduit
- ✅ Couche Dense intermédiaire (256 neurones)

## Améliorations Possibles

Pour aller au-delà de 90.79% :
- Augmenter la taille des images (128×128 ou 160×160)
- Tester d'autres architectures (EfficientNetV2, ResNet50V2)
- Progressive unfreezing (dégeler progressivement plus de couches)
- Augmenter le nombre d'epochs
- Learning rate scheduling

## Auteur

**Abderrahmane Ghomed** - Élève en C25  
Projet réalisé dans le cadre du cursus Holberton School - Machine Learning

## Licence

Ce projet est à usage éducatif uniquement.