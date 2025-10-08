# CNN - Implémentation From Scratch

Implémentation complète des opérations fondamentales d'un réseau de neurones convolutif (CNN) avec NumPy et Keras.

---

## 📋 Exercices réalisés

### 1. Convolution Forward (`0-conv_forward.py`)
Propagation avant d'une couche convolutionnelle.

**Concepts :**
- Padding : `same` (conserve dimensions) vs `valid` (réduit dimensions)
- Stride : pas de déplacement du filtre
- Formule : `output_size = (input_size + 2*pad - kernel_size) // stride + 1`

**Optimisation :** Vectorisation NumPy → 50x plus rapide !

```python
# Traite toutes les images simultanément
region = A_prev_padded[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
conv_value = np.sum(region * kernel, axis=(1, 2, 3)) + b[0, 0, 0, k]
A[:, i, j, k] = activation(conv_value)
```

**Performance :**
- Input : `(50000, 28, 28, 1)`
- Output : `(50000, 26, 26, 2)` avec `padding='valid'`

---

### 2. Pooling Forward (`1-pool_forward.py`)
Réduction de dimensions par max ou average pooling.

**Max Pooling :** Garde la valeur maximale → features les plus fortes  
**Average Pooling :** Calcule la moyenne → lisse les features

```python
if mode == 'max':
    A[:, i, j, :] = np.max(region, axis=(1, 2))
elif mode == 'avg':
    A[:, i, j, :] = np.mean(region, axis=(1, 2))
```

**Performance :**
- Input : `(50000, 28, 28, 2)`
- Output : `(50000, 14, 14, 2)` avec kernel `(2, 2)` et stride `(2, 2)`

---

### 3. Convolution Backward (`2-conv_backward.py`)
Rétropropagation pour calculer les gradients.

**Retourne :**
- `dA_prev` : gradients vers la couche précédente
- `dW` : gradients des poids (pour mise à jour)
- `db` : gradients des biais

**Formules clés :**
```python
# Gradient des biais
db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

# Gradient des poids
dW[:, :, :, k] += np.sum(region * dZ_slice, axis=0)

# Gradient de l'entrée
dA_prev_padded[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :] += W_slice * dZ_slice
```

**Intuition :** `dZ` indique "qui est responsable de l'erreur" et on remonte pour distribuer cette responsabilité.

---

### 4. Pooling Backward (`3-pool_backward.py`)
Rétropropagation du pooling.

**Max Pooling :** Gradient va uniquement au maximum
```python
mask = (region == max_val).astype(float)
dA_prev += mask * dA[:, i, j, k][:, np.newaxis, np.newaxis]
```

**Average Pooling :** Gradient distribué également
```python
dA_prev += dA[:, i, j, k][:, np.newaxis, np.newaxis] / (kh * kw)
```

**Logique :**
- **Max** : Seul le max a contribué → seul lui reçoit le gradient
- **Average** : Tous ont contribué également → distribution égale

---

### 5. LeNet-5 (Keras) (`5-lenet5.py`)
CNN complet pour classification MNIST.

**Architecture :**
```
Input (28×28×1)
    ↓
Conv2D (6 filtres 5×5, same) + ReLU → (28×28×6)
    ↓
MaxPool (2×2, stride 2) → (14×14×6)
    ↓
Conv2D (16 filtres 5×5, valid) + ReLU → (10×10×16)
    ↓
MaxPool (2×2, stride 2) → (5×5×16)
    ↓
Flatten → Dense(120) → Dense(84) → Dense(10, softmax)
```

**Code complet :**
```python
from tensorflow import keras as K

def lenet5(X):
    initializer = K.initializers.HeNormal(seed=0)
    
    conv1 = K.layers.Conv2D(6, (5,5), padding='same', 
                            activation='relu', 
                            kernel_initializer=initializer)(X)
    pool1 = K.layers.MaxPooling2D((2,2), strides=(2,2))(conv1)
    
    conv2 = K.layers.Conv2D(16, (5,5), padding='valid',
                            activation='relu',
                            kernel_initializer=initializer)(pool1)
    pool2 = K.layers.MaxPooling2D((2,2), strides=(2,2))(conv2)
    
    flatten = K.layers.Flatten()(pool2)
    fc1 = K.layers.Dense(120, activation='relu', 
                         kernel_initializer=initializer)(flatten)
    fc2 = K.layers.Dense(84, activation='relu',
                         kernel_initializer=initializer)(fc1)
    output = K.layers.Dense(10, activation='softmax',
                           kernel_initializer=initializer)(fc2)
    
    model = K.Model(inputs=X, outputs=output)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
```

---

## 🧠 Concepts clés

### Convolution
Applique un filtre sur l'image pour extraire des features (contours, textures, formes).

### Padding
- **same** : conserve les dimensions (ajoute des zéros)
- **valid** : aucun padding (dimensions réduites)

### Stride
Pas de déplacement du filtre. `stride > 1` → réduit les dimensions.

### Pooling
Réduit dimensions sans paramètres apprenables. Apporte une invariance aux translations.

### Backpropagation
Calcul des gradients pour mettre à jour les poids : `W = W - learning_rate * dW`

---

## 📊 Résultats LeNet-5 sur MNIST

| Métrique | Valeur |
|----------|--------|
| **Accuracy (train)** | 99.24% |
| **Accuracy (validation)** | 98.70% |
| **Loss finale (train)** | 0.024 |
| **Loss finale (validation)** | 0.054 |
| **Temps d'entraînement** | ~53s (5 epochs) |
| **Nombre de paramètres** | ~61,000 |

**Entraînement détaillé :**
```
Epoch 1/5: accuracy: 94.56% → val_accuracy: 97.94%
Epoch 2/5: accuracy: 98.16% → val_accuracy: 98.34%
Epoch 3/5: accuracy: 98.68% → val_accuracy: 98.48%
Epoch 4/5: accuracy: 99.01% → val_accuracy: 98.74%
Epoch 5/5: accuracy: 99.24% → val_accuracy: 98.70%
```

**Exemple de prédiction :**
Pour un chiffre "3", le modèle prédit :
```python
[3.02e-16, 4.20e-11, 8.19e-13, 1.00, 1.12e-16, ...]
                                  ↑
                    Probabilité 100% pour la classe 3
```

---

## 💡 Points clés

### 1. Vectorisation NumPy

```python
# ❌ Lent : boucle sur chaque image
for img in range(m):
    result[img] = operation(data[img])

# ✅ 50x plus rapide : traite toutes les images simultanément
result = operation(data)
```

### 2. Architecture CNN moderne

```
[Convolution → Activation → Pooling] × N blocs
    ↓
Flatten (aplatir en vecteur)
    ↓
Dense layers (fully connected)
    ↓
Softmax (probabilités de classification)
```

### 3. From scratch vs Keras

| Aspect | NumPy (from scratch) | Keras |
|--------|---------------------|-------|
| **Compréhension** | Profonde | Utilisation |
| **Contrôle** | Total | Abstrait |
| **Temps de dev** | Long | Rapide |
| **Performance** | À optimiser | Optimisé auto |
| **Debugging** | Difficile | Facile |

**Conclusion :** Les deux approches sont complémentaires !

---

## 🚀 Compétences acquises

- [x] Implémentation from scratch des opérations CNN avec NumPy
- [x] Optimisation avec vectorisation (50x speedup)
- [x] Compréhension profonde de la backpropagation
- [x] Construction d'architectures complètes avec Keras
- [x] Entraînement et évaluation de modèles
- [x] Debugging d'erreurs de dimensions et de syntaxe
- [x] Obtention de >98% accuracy sur MNIST

---

## 📐 Formules importantes

**Dimensions de sortie (convolution) :**
```
output_h = (input_h + 2*pad_h - kernel_h) // stride_h + 1
output_w = (input_w + 2*pad_w - kernel_w) // stride_w + 1
```

**Padding pour "same" (avec stride=1) :**
```
pad_h = (kernel_h - 1) // 2
pad_w = (kernel_w - 1) // 2
```

**Dimensions de sortie (pooling) :**
```
output_h = (input_h - kernel_h) // stride_h + 1
output_w = (input_w - kernel_w) // stride_w + 1
```

---

## 🎯 Pour aller plus loin

### Architectures modernes à explorer
- **VGG** : Convolutions 3×3 empilées
- **ResNet** : Skip connections (connexions résiduelles)
- **Inception** : Convolutions multi-échelles parallèles
- **EfficientNet** : Scaling optimal des dimensions

### Techniques avancées
- **Data Augmentation** : rotation, flip, zoom
- **Batch Normalization** : stabilise l'entraînement
- **Dropout** : régularisation anti-overfitting
- **Transfer Learning** : réutiliser modèles pré-entraînés
- **Learning Rate Scheduling** : ajuster le taux d'apprentissage

### Applications réelles
- Classification d'images
- Détection d'objets (YOLO, Faster R-CNN)
- Segmentation sémantique (U-Net, Mask R-CNN)
- Style transfer
- Super-resolution

---

## 📚 Ressources

- **Paper LeNet-5** : "Gradient-Based Learning Applied to Document Recognition" (LeCun et al., 1998)
- **Cours deeplearning.ai** : CNN Specialization
- **CS231n Stanford** : Convolutional Neural Networks for Visual Recognition
- **Documentation** : https://keras.io/

---

## 📂 Structure du projet

```
supervised_learning/cnn/
├── 0-conv_forward.py          # Convolution forward
├── 1-pool_forward.py           # Pooling forward
├── 2-conv_backward.py          # Convolution backward
├── 3-pool_backward.py          # Pooling backward
├── 5-lenet5.py                 # LeNet-5 avec Keras
└── README.md                   # Ce fichier
```

---

*Implémentation complète du forward et backward pass d'un CNN, avec validation sur MNIST atteignant 98.7% d'accuracy.*

**Technologies :** Python, NumPy, TensorFlow/Keras  
**Dataset :** MNIST (chiffres manuscrits 0-9)  
**Date :** Octobre 2025