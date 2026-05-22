# TP sur la Régularisation en Deep Learning

## 📋 Objectif

Ce TP explore les techniques de régularisation pour réduire l'overfitting (sur-apprentissage) dans les réseaux de neurones, appliqué à un problème de prédiction de positions de tir pour l'équipe de football française.

## 🎯 Problème

La Fédération Française de Football veut savoir où le gardien doit tirer le ballon pour maximiser les chances que les joueurs français le touchent de la tête.

- **Points bleus** = succès français
- **Points rouges** = échec (adversaire touche le ballon)

## 🏗️ Architecture du Réseau

Réseau de neurones à 3 couches :
- **Entrée** → 20 neurones (ReLU) → 3 neurones (ReLU) → 1 neurone (Sigmoid) → **Sortie**

## 📊 Résultats Comparatifs

| Modèle | Train Accuracy | Test Accuracy |
|--------|----------------|---------------|
| Sans régularisation | 94.8% | 91.5% |
| L2 (λ=0.7) | 93.8% | 93% |
| Dropout (keep_prob=0.86) | 92.9% | 95% |

## 🔬 Techniques de Régularisation

### 1. Régularisation L2

**Principe** : Pénaliser les grands poids dans la fonction de coût.

**Formule du coût** :
```
J = erreur_cross_entropy + (λ/2m) × Σ||W||²
```

**Effet** :
- Force les poids à rester petits
- Rend le modèle plus simple et stable
- Frontière de décision plus lisse

**Modification des gradients** :
```
dW = dW_original + (λ/m) × W
```

### 2. Dropout

**Principe** : Désactiver aléatoirement des neurones pendant l'entraînement.

**Processus (4 étapes)** :
1. Créer un masque aléatoire D
2. Convertir en 0 ou 1 : `D = (D < keep_prob).astype(int)`
3. Désactiver : `A = A × D`
4. Mettre à l'échelle : `A = A / keep_prob`

**Pourquoi le scaling ?**
Pour maintenir la même valeur attendue des activations après désactivation aléatoire.

**Effet** :
- Force le réseau à ne pas dépendre de neurones spécifiques
- Apprend des caractéristiques redondantes et robustes
- Meilleure généralisation

## 📈 Observations Clés

1. **La régularisation réduit la performance sur le train**
   - Normal : elle limite la capacité à sur-apprendre
   - Sans régularisation, le modèle peut "tricher" en mémorisant

2. **Mais elle améliore la performance sur le test**
   - C'est l'objectif : meilleure généralisation
   - Le modèle apprend le pattern général, pas le bruit

3. **Dropout donne le meilleur résultat**
   - 95% sur le test = meilleure généralisation
   - Très efficace pour prévenir l'overfitting

## 🎓 Concepts Clés

- **Overfitting** : Modèle apprend trop bien les données d'entraînement, mémorise le bruit
- **Régularisation L2** : Pénalité sur les grands poids → modèle plus simple
- **Dropout** : Désactivation aléatoire de neurones → caractéristiques robustes
- **Généralisation** : Capacité du modèle à performer sur de nouvelles données

## 💡 Conclusion

La régularisation est essentielle pour construire des modèles qui généralisent bien. Même si elle réduit la performance sur les données d'entraînement, elle améliore significativement la performance sur de nouvelles données - c'est ce qui compte dans la pratique !

## 🔧 Fichiers

- `Regularization_v2a.ipynb` : Notebook principal avec le TP complet
- `reg_utils.py` : Fonctions utilitaires pour le réseau de neurones
- `testCases.py` : Tests unitaires pour les fonctions implémentées
- `datasets/` : Données d'entraînement et de test
- `images/` : Images pour le notebook

## 🚀 Auteur

TP réalisé dans le cadre du cours de Deep Learning sur la régularisation.
