# README - Convolutions et Pooling

Série d'exercices complète sur les opérations fondamentales des réseaux de neurones convolutifs (CNNs).

---

## Vue d'ensemble du projet

Ce projet implémente from scratch (sans utiliser `np.convolve`) toutes les opérations de base d'un CNN :
- Convolutions avec différents modes de padding
- Gestion du stride
- Support des images multi-canaux
- Multiple kernels
- Pooling (max et average)

---

## Exercices réalisés

### 0. Convolution Valid (Grayscale)
**Fichier :** `0-convolve_grayscale_valid.py`

Première implémentation de convolution sans padding.

**Concepts clés :**
- Convolution "valid" = aucun padding
- L'image rétrécit : `output_size = input_size - kernel_size + 1`
- Exemple : 28×28 → 26×26 avec kernel 3×3

**Code :**
```python
def convolve_grayscale_valid(images, kernel):
    # Images : (m, h, w)
    # Kernel : (kh, kw)
    # Output : (m, output_h, output_w)
```

---

### 1. Convolution Same (Grayscale)
**Fichier :** `1-convolve_grayscale_same.py`

Ajout du padding automatique pour conserver la taille de l'image.

**Concepts clés :**
- Padding calculé automatiquement : `pad = (kernel_size - 1) // 2`
- Output a la même taille que l'input
- Création manuelle du padding avec `np.zeros()`

**Padding :**
```python
padded_images = np.zeros((m, h+2*pad_h, w+2*pad_w))
padded_images[:, pad_h:pad_h+h, pad_w:pad_w+w] = images
```

---

### 2. Convolution Custom Padding (Grayscale)
**Fichier :** `2-convolve_grayscale_padding.py`

Padding défini par l'utilisateur.

**Concepts clés :**
- Padding arbitraire : tuple `(ph, pw)`
- Dimensions de sortie : `((h + 2*ph - kh) // stride) + 1`
- Contrôle total sur la taille de sortie

---

### 3. Convolution avec Stride (Grayscale)
**Fichier :** `3-convolve_grayscale.py`

Fonction générique combinant tous les modes de padding + stride.

**Concepts clés :**
- Stride = "pas" de déplacement du kernel
- `stride=(2,2)` divise les dimensions par ~2
- Indices multipliés par stride : `i*sh` au lieu de `i`

**Gestion du padding :**
```python
if padding == 'same':
    ph = (kh - 1) // 2
elif padding == 'valid':
    ph, pw = 0, 0
else:  # tuple (ph, pw)
    ph, pw = padding
```

---

### 4. Convolution avec Canaux (RGB)
**Fichier :** `4-convolve_channels.py`

Extension aux images couleur.

**Concepts clés :**
- Images : `(m, h, w, c)` - c canaux (RGB = 3)
- Kernel : `(kh, kw, c)` - couvre tous les canaux
- Output : `(m, output_h, output_w)` - UN canal (somme sur tous les canaux)
- Somme sur 3 axes : `axis=(1, 2, 3)`

**Extraction avec canaux :**
```python
region = padded_images[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
output[:, i, j] = (region * kernel).sum(axis=(1, 2, 3))
```

---

### 5. Convolution Multiple Kernels
**Fichier :** `5-convolve.py`

Application de plusieurs filtres simultanément.

**Concepts clés :**
- Kernels : `(kh, kw, c, nc)` - nc kernels différents
- Output : `(m, output_h, output_w, nc)` - nc feature maps
- 3 boucles autorisées : une pour chaque kernel

**Boucle sur kernels :**
```python
for k in range(nc):
    current_kernel = kernels[:, :, :, k]
    for i in range(output_h):
        for j in range(output_w):
            region = padded_images[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
            output[:, i, j, k] = (region * current_kernel).sum(axis=(1, 2, 3))
```

---

### 6. Pooling
**Fichier :** `6-pool.py`

Réduction de dimensionnalité par max ou average pooling.

**Concepts clés :**
- Pas de padding en pooling (standard)
- Pas de multiplication, juste sélection (max) ou moyenne
- S'applique indépendamment sur chaque canal
- Max pooling : `np.max(region, axis=(1, 2))`
- Avg pooling : `np.mean(region, axis=(1, 2))`

**Différence avec convolution :**
```python
# Convolution : multiplication + somme
output = (region * kernel).sum()

# Pooling : sélection simple
output = np.max(region, axis=(1, 2))  # ou np.mean()
```

---

## Concepts fondamentaux maîtrisés

### 1. Le Kernel (Filtre)
Petite grille de nombres qui détecte des features :
- Contours (horizontaux, verticaux, diagonaux)
- Textures
- Couleurs spécifiques
- Formes géométriques

### 2. Le Padding
Ajout de pixels (généralement zéros) autour de l'image :
- **Valid** : pas de padding → image rétrécit
- **Same** : padding auto → même taille
- **Custom** : padding défini → contrôle total

### 3. Le Stride
Pas de déplacement du kernel :
- `stride=1` : pixel par pixel (standard)
- `stride=2` : saute un pixel → réduit de ~50%
- Utilisé pour réduire les dimensions rapidement

### 4. Les Canaux
- **Entrée RGB** : 3 canaux (Rouge, Vert, Bleu)
- **Convolution** : somme sur tous les canaux d'entrée
- **Multiple kernels** : produit plusieurs canaux de sortie

### 5. Le Pooling
Réduction de taille sans paramètres apprenables :
- **Max pooling** : garde la valeur maximale (plus courant)
- **Average pooling** : fait la moyenne
- Réduit les dimensions et la complexité
- Apporte une invariance aux petites translations

---

## Architecture CNN typique

```
Input: (224, 224, 3)
    ↓
Conv + ReLU: 32 kernels 3×3, same, stride=1 → (224, 224, 32)
    ↓
Conv + ReLU: 32 kernels 3×3, same, stride=1 → (224, 224, 32)
    ↓
MaxPool: 2×2, stride=2 → (112, 112, 32)
    ↓
Conv + ReLU: 64 kernels 3×3, same, stride=1 → (112, 112, 64)
    ↓
Conv + ReLU: 64 kernels 3×3, same, stride=1 → (112, 112, 64)
    ↓
MaxPool: 2×2, stride=2 → (56, 56, 64)
    ↓
... (plusieurs couches similaires)
    ↓
Flatten + Dense layers → Classification
```

---

## Formules importantes

**Dimensions de sortie (convolution) :**
```
output_h = ((h + 2*ph - kh) // sh) + 1
output_w = ((w + 2*pw - kw) // sw) + 1
```

**Padding pour "same" :**
```
ph = (kh - 1) // 2
pw = (kw - 1) // 2
```

**Dimensions de sortie (pooling) :**
```
output_h = (h - kh) // sh + 1
output_w = (w - kw) // sw + 1
```

---

## Compétences développées

- Manipulation de tenseurs NumPy 3D et 4D
- Slicing avancé et broadcasting
- Optimisation avec vectorisation (pas de boucles inutiles)
- Compréhension profonde des opérations CNN
- Implémentation from scratch sans librairies haut niveau

---

## Progression pédagogique

1. **Grayscale valid** → Mécanique de base
2. **Grayscale same** → Ajout du padding
3. **Custom padding** → Flexibilité
4. **Stride** → Fonction complète grayscale
5. **Channels** → Images couleur
6. **Multiple kernels** → Convolution complète comme dans les vrais CNNs
7. **Pooling** → Opération complémentaire

Chaque exercice construit sur le précédent, ajoutant progressivement de la complexité jusqu'à avoir une implémentation complète d'une couche de convolution professionnelle.