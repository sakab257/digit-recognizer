# Reconnaissance de Chiffres Manuscrits IA

Un projet de reconnaissance de chiffres manuscrits utilisant un réseau de neurones convolutionnel (CNN) avec une interface graphique moderne et intuitive développée en Python.

## Description

Ce projet permet de dessiner des chiffres à la main sur une interface graphique épurée et d'obtenir une prédiction en temps réel grâce à un modèle de deep learning entraîné sur le dataset MNIST. Le système affiche la prédiction principale, le niveau de confiance et les alternatives possibles dans une interface moderne et professionnelle.

## Fonctionnalités

- **Interface graphique moderne** : Design épuré avec canvas de dessin haute résolution
- **Prédiction en temps réel** : Classification instantanée des chiffres dessinés
- **Analyse de confiance** : Affichage du niveau de certitude avec indicateurs visuels
- **Alternatives proposées** : Top 3 des prédictions possibles avec leurs probabilités
- **Prétraitement automatique** : Normalisation et redimensionnement intelligents
- **Design responsive** : Interface adaptée et centrée pour une meilleure expérience

## Technologies Utilisées

- **Python 3.7+**
- **TensorFlow/Keras** : Pour l'entraînement et l'inférence du modèle
- **Tkinter** : Interface graphique native avec design moderne
- **OpenCV** : Traitement d'image avancé
- **NumPy** : Calculs numériques optimisés
- **PIL/Pillow** : Manipulation d'images
- **Matplotlib** : Visualisation des métriques d'entraînement

## Installation

### Prérequis

```bash
pip install tensorflow keras opencv-python pillow numpy matplotlib
```

### Installation rapide

1. Clonez ou téléchargez le projet
2. Installez les dépendances :
```bash
pip install tensorflow keras opencv-python pillow numpy matplotlib
```
3. Lancez l'entraînement du modèle (si nécessaire) :
```bash
python train_digit_recognizer.py
```
4. Lancez l'interface graphique :
```bash
python gui_digit_recognizer.py
```

## Utilisation

### Interface Graphique

```bash
python gui_digit_recognizer.py
```

**Guide d'utilisation de l'interface moderne :**

1. **Dessinez** un chiffre (0-9) sur le canvas blanc avec votre souris
2. **Cliquez** sur le bouton "Analyser" pour obtenir la classification
3. **Observez** les résultats dans le panneau centralisé :
   - Prédiction principale avec code couleur dynamique
   - Niveau de confiance en pourcentage avec indicateur visuel
   - Top 2 des alternatives possibles
4. **Cliquez** sur "Effacer" pour nettoyer le canvas et recommencer

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

## Architecture du Projet

```
digit-recognizer/
│
├── train_digit_recognizer.py    # Script d'entraînement du modèle
├── gui_digit_recognizer.py      # Interface graphique moderne
├── mnist_improved.h5            # Modèle entraîné (généré)
├── training_history.png         # Graphiques de performance (généré)
└── README.md                   # Documentation
```

## Interface Utilisateur

### Design Moderne

L'interface a été entièrement repensée avec :

- **Thème clair professionnel** : Palette de couleurs moderne (#f8f9fa, #007acc)
- **Typographie Segoe UI** : Police moderne et lisible
- **Canvas haute résolution** : 340x340 pixels avec bordure élégante
- **Boutons stylisés** : Design flat avec effets hover
- **Panneau de résultats centré** : Affichage organisé et lisible
- **Indicateurs visuels** : Couleurs adaptatives selon la confiance

### Expérience Utilisateur

- Canvas avec curseur précis et highlight au focus
- Boutons avec feedback visuel et curseur interactif
- Textes centrés et hiérarchisés pour une lecture optimale
- Instructions claires et discrètes
- Design responsive et professionnel

## Architecture du Modèle

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

## Performances

- **Précision sur le test MNIST** : ~99.5%
- **Temps d'inférence** : <100ms par prédiction
- **Robustesse** : Gestion automatique des variations de style d'écriture

Le modèle a été spécialement optimisé pour bien fonctionner avec des dessins faits à la souris, incluant :
- Prétraitement adaptatif (inversion, centrage, redimensionnement)
- Gestion des ratio d'aspect variables
- Normalisation automatique des intensités

## Fonctionnalités Techniques

### Prétraitement Intelligent
- **Capture directe du canvas** : Sans dépendance à Ghostscript
- **Inversion automatique** : Conversion blanc sur noir (format MNIST)
- **Détection des contours** : Isolation automatique du chiffre dessiné
- **Centrage adaptatif** : Positionnement optimal dans l'image 28x28
- **Préservation du ratio** : Évite la déformation des chiffres

### Interface Moderne
- **Design épuré** : Interface professionnelle sans éléments superflus
- **Feedback visuel avancé** : Codes couleur et indicateurs de confiance
- **Responsive design** : Centrage automatique et adaptation de l'affichage
- **Accessibilité** : Couleurs contrastées et typographie lisible

## Conseils d'Utilisation

Pour obtenir les meilleures prédictions :

1. **Dessinez au centre** du canvas
2. **Utilisez une taille suffisante** (ni trop petit, ni trop grand)
3. **Tracez clairement** les contours du chiffre
4. **Évitez les traits parasites** dans les coins
5. **Respectez les formes standards** des chiffres

## Résolution de Problèmes

**Le modèle n'est pas trouvé :**
```
Solution: Lancez d'abord train_digit_recognizer.py pour créer le modèle
```

**Erreur d'importation :**
```
Solution: Vérifiez que toutes les dépendances sont installées
pip install tensorflow keras opencv-python pillow numpy matplotlib
```

**Prédictions incorrectes :**
```
Solution: Assurez-vous de dessiner clairement au centre du canvas
```

## Améliorations Futures

- [ ] Support de plusieurs chiffres simultanés
- [ ] Reconnaissance de lettres
- [ ] Sauvegarde des dessins
- [ ] Mode d'entraînement en ligne
- [ ] Export du modèle vers d'autres formats (ONNX, TensorFlow Lite)
- [ ] Interface web responsive
- [ ] Mode sombre/clair

## Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :

1. Fork le projet
2. Créer une branche pour votre fonctionnalité
3. Commiter vos changements
4. Pusher vers la branche
5. Ouvrir une Pull Request

## Licence

Ce projet est sous licence MIT.

## Remerciements

- Dataset MNIST pour l'entraînement
- Communauté TensorFlow/Keras pour les outils
- Communauté open source pour les bibliothèques utilisées

---

**Développé avec passion en Python**