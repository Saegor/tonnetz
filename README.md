# Tonnetz Acoustique Orthogonal

Le Tonnetz Acoustique Orthogonal (TAO) est un projet innovant d'analyse audio en temps réel qui extrait des notes musicales d'un flux audio et les affiche dans une représentation visuelle basée sur le concept du Tonnetz. Inspiré par la physique des harmoniques naturelles et le tempérament égal des 12 tons, ce projet propose une approche originale de visualisation harmonique fondée sur les 3ᵉ et 5ᵉ harmoniques.

## Table des Matières

- [Introduction](#introduction)
- [Fonctionnalités](#fonctionnalités)
- [Présentation du Tonnetz Acoustique](#présentation-du-tonnetz-acoustique)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Architecture du Projet](#architecture-du-projet)
- [Licence](#licence)

## Introduction

Ce projet a pour objectif d'analyser un flux audio en temps réel, d'extraire les notes (via la détection de pics dans le spectre sonore) et de les utiliser pour alimenter une représentation graphique du Tonnetz. La grille harmonique ainsi construite repose sur une organisation des intervalles basée sur les rapports des harmoniques naturelles (3ᵉ et 5ᵉ), offrant une visualisation inédite et fidèle à la réalité physique des sons.

## Fonctionnalités

- **Analyse audio en temps réel** : Utilisation de PyAudio pour capter le son via le microphone et extraction des caractéristiques sonores.
- **Détection des notes** : Transformation des signaux audio via la FFT et identification des pics d'amplitude dépassant un seuil défini.
- **Visualisation interactive** : Affichage du Tonnetz à l'aide de Matplotlib, avec superposition dynamique des notes détectées.
- **Gestion multi-thread** : Séparation du flux audio et de l'affichage pour une meilleure performance et fluidité.
- 
## Présentation du Tonnetz Acoustique

Le Tonnetz (ou « réseau tonal ») est une représentation géométrique des relations entre les notes, historiquement utilisée pour décrire les connexions harmoniques. Dans ce projet, la grille du Tonnetz est définie par :

- Un calcul spécifique de la classe de hauteur (pitch class) basé sur une combinaison linéaire (7 * i + 4 * j) modulo 12.
- Une représentation orthogonale qui permet d’assigner à chaque case une note, avec un affichage sous forme de grille.
- La superposition des notes détectées en temps réel, avec un ajustement de la position (offset) en fonction des déviations en cents, offrant ainsi une visualisation fine de la justesse de l'intonation.

## Installation

Pour lancer ce projet, vous devez disposer de Python 3 et installer les dépendances nécessaires via pip :

```bash
pip install numpy scipy matplotlib pyaudio
```

Assurez-vous également que votre environnement audio (microphone, carte son) est correctement configuré.

## Utilisation

Après installation, lancez le script principal :

```bash
python main.py
```

Le programme capte alors le flux audio en temps réel, extrait les notes et met à jour dynamiquement la représentation graphique du Tonnetz.

## Architecture du Projet

Le projet est structuré en plusieurs parties :

- **Analyse Audio**  
  Utilisation de PyAudio pour la capture et conversion du signal, suivi d’un traitement par FFT pour en extraire le spectre et détecter les pics d'amplitude.

- **Traitement des Données**  
  Transformation des pics en notes MIDI, avec conversion des fréquences en valeurs MIDI basées sur la référence 440 Hz.

- **Visualisation**  
  Création d’une grille Tonnetz grâce à Matplotlib. La couleur et la position des éléments (notes) sont gérées en fonction des paramètres définis pour la représentation harmonique.

- **Gestion Multi-thread**  
  Séparation du thread d’écoute audio et du thread d’affichage pour assurer une performance fluide et réactive.

## Licence

Ce projet est distribué sous licence GNU General Public License v3.0 (GNU GPL v3).
