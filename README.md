# neurofeedback-engagement-decoding
DESU tutored project

## Introduction
Code du projet (DESU Data Science) pour le décodage d’états cognitifs à partir d’activités multi-neurones.
Le pipeline réalise un **décodage sliding-window** via **régression logistique Elastic Net (SGD)** avec **z-score** et **validation croisée stratifiée**.

## Modules
- `neuron_analysis/main.py` : fonctions d’alignement & builder (essais × neurones)
- `neuron_analysis/data_loading.py` : utilitaires de chargement
- `neuron_analysis/decoding.py` : décodage ElasticNet en fenêtres glissantes
- `neuron_analysis/stability.py` : mesures de stabilité des poids (β)