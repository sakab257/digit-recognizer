# Reconnaissance de Chiffres Manuscrits 🔢

Un projet de reconnaissance de chiffres manuscrits utilisant un réseau de neurones convolutionnel (CNN) et une interface graphique intuitive développée en Python.

## 📋 Description

Ce projet permet de dessiner des chiffres à la main sur une interface graphique et d'obtenir une prédiction en temps réel grâce à un modèle de deep learning entraîné sur le dataset MNIST. Le système affiche non seulement la prédiction principale mais aussi le niveau de confiance et les alternatives possibles.

## ✨ Fonctionnalités

- **Interface graphique intuitive** : Canvas de dessin avec pinceau adaptatif
- **Prédiction en temps réel** : Classification instantanée des chiffres dessinés
- **Analyse de confiance** : Affichage du niveau de certitude de la prédiction
- **Alternatives proposées** : Top 3 des prédictions possibles avec leurs probabilités
- **Prétraitement automatique** : Normalisation et redimensionnement automatiques
- **Indicateurs visuels** : Code couleur selon le niveau de confiance

## 🛠️ Technologies Utilisées

- **Python 3.7+**
- **TensorFlow/Keras** : Pour l'entraînement et l'inférence du modèle
- **Tkinter** : Interface graphique native
- **OpenCV** : Traitement d'image
- **NumPy** : Calculs numériques
- **PIL/Pillow** : Manipulation d'images
- **Matplotlib** : Visualisation des métriques d'entraînement

## 📦 Installation

### Prérequis

```bash
pip install tensorflow keras opencv-python pillow numpy matplotlib pywin32
```

### Installation rapide

1. Clonez ou téléchargez le projet
2. Installez les dépendances :
```bash
pip install -r requirements.txt
```
3. Lancez l'entraînement du modèle (optionnel si vous n'avez pas le modèle pré-entraîné) :
```bash
python train_digit_recognizer.py
```
4. Lancez l'interface graphique :
```bash
python gui_digit_recognizer.py
```

## 🚀 Utilisation

### Entraînement du Modèle

```bash
python train_digit_recognizer.py
```

Le script d'entraînement :
- Charge automatiquement le dataset MNIST
- Applique l'augmentation de données pour améliorer la généralisation
- Utilise des techniques avancées (BatchNormalization, Dropout, Callbacks)
- Sauvegarde le meilleur modèle sous `mnist_improved.h5`
- Génère des graphiques de performance (`training_history.png`)

### Interface Graphique

```bash
python gui_digit_recognizer.py
```

**Comment utiliser l'interface :**

1. **Dessinez** un chiffre (0-9) sur le canvas blanc avec votre souris
2. **Cliquez** sur "Prédire" pour obtenir la classification
3. **Observez** les résultats :
   - Prédiction principale avec code couleur
   - Niveau de confiance en pourcentage
   - Top 2 des alternatives possibles
4. **Cliquez** sur "Effacer" pour recommencer

## 🏗️ Architecture du Projet

```
projet/
│
├── train_digit_recognizer.py    # Script d'entraînement du modèle
├── gui_digit_recognizer.py      # Interface graphique principale
├── mnist_improved.h5            # Modèle entraîné (généré)
├── training_history.png         # Graphiques de performance (généré)
├── requirements.txt             # Dépendances Python
└── README.md                   # Documentation
```

## 🧠 Architecture du Modèle

Le modèle utilise une architecture CNN optimisée :

```
Entrée (28x28x1) 
    ↓
Bloc Conv1: Conv2D(32) + BatchNorm + Conv2D(32) + BatchNorm + MaxPool + Dropout
    ↓
Bloc Conv2: Conv2D(64) + BatchNorm + Conv2D(64) + BatchNorm + MaxPool + Dropout
    ↓
Flatten + Dense(256) + BatchNorm + Dropout
    ↓
Dense(128) + BatchNorm + Dropout
    ↓
Dense(10, softmax) → Prédiction finale
```

**Techniques d'optimisation :**
- **Batch Normalization** : Stabilise l'entraînement
- **Dropout** : Prévient le surajustement
- **Data Augmentation** : Améliore la généralisation
- **Early Stopping** : Évite le surentraînement
- **Learning Rate Scheduling** : Optimise la convergence

## 📊 Performances

- **Précision sur le test MNIST** : ~99.5%
- **Temps d'inférence** : <100ms par prédiction
- **Robustesse** : Gestion automatique des variations de style d'écriture

Le modèle a été spécialement optimisé pour bien fonctionner avec des dessins faits à la souris, incluant :
- Prétraitement adaptatif (inversion, centrage, redimensionnement)
- Gestion des ratio d'aspect variables
- Normalisation automatique des intensités

## 🔧 Fonctionnalités Techniques

### Prétraitement Intelligent
- **Inversion automatique** : Conversion blanc sur noir (format MNIST)
- **Détection des contours** : Isolation automatique du chiffre dessiné
- **Centrage adaptatif** : Positionnement optimal dans l'image 28x28
- **Préservation du ratio** : Évite la déformation des chiffres

### Interface Utilisateur
- **Dessin fluide** : Lignes continues avec lissage
- **Feedback visuel** : Codes couleur selon la confiance
- **Conseils intégrés** : Instructions d'utilisation dans l'interface

## 🎯 Conseils d'Utilisation

Pour obtenir les meilleures prédictions :

1. **Dessinez au centre** du canvas
2. **Utilisez une taille suffisante** (ni trop petit, ni trop grand)
3. **Tracez clairement** les contours du chiffre
4. **Évitez les traits parasites** dans les coins
5. **Respectez les formes standards** des chiffres

## 🐛 Résolution de Problèmes

**Le modèle n'est pas trouvé :**
```
Solution: Lancez d'abord train_digit_recognizer.py pour créer le modèle
```

**Erreur d'importation :**
```
Solution: Vérifiez que toutes les dépendances sont installées
pip install -r requirements.txt
```

**Prédictions incorrectes :**
```
Solution: Assurez-vous de dessiner clairement au centre du canvas
```

## 📈 Améliorations Futures

- [ ] Support de plusieurs chiffres simultanés
- [ ] Reconnaissance de lettres
- [ ] Sauvegarde des dessins
- [ ] Mode d'entraînement en ligne
- [ ] Export du modèle vers d'autres formats (ONNX, TensorFlow Lite)
- [ ] Interface web avec Flask/Django

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :

1. Fork le projet
2. Créer une branche pour votre fonctionnalité
3. Commiter vos changements
4. Pusher vers la branche
5. Ouvrir une Pull Request

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 🙏 Remerciements

- Dataset MNIST pour l'entraînement
- Communauté TensorFlow/Keras pour les outils
- Communauté open source pour les bibliothèques utilisées

---

**Développé avec ❤️ en Python**