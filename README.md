# Lisez-moi

## Aperçu

Ce dépôt contient des scripts Python conçus pour l'analyse de texte et le nettoyage des données dans le cadre de tâches de Traitement Automatique du Langage Naturel (TALN), ici spécifiquement pour le français.

Les scripts sont organisés en trois fichiers principaux :

1. **fonctions_cleaning.py** : Ce script contient des fonctions dédiées au nettoyage des données. Il se concentre sur des tâches de prétraitement telles que la gestion des valeurs manquantes et la standardisation des formats à travers les jeux de données.

2. **fonctions_cleaning_text.py** : Ce script étend les fonctionnalités de `fonctions_cleaning.py` en incluant des fonctions supplémentaires spécifiques au nettoyage de texte, comme la suppression des stopwords, la normalisation du texte et le stemming/lémmatisation.

3. **fonctions_analyse_text.py** : Ce script fournit des fonctions pour l'analyse des données textuelles, y compris la tokenisation, l'analyse de fréquence, et d'autres opérations courantes en TALN.

## Description des Fichiers

### 1. fonctions_cleaning.py

Ce fichier est dédié au nettoyage des données brutes afin de les préparer pour l'analyse. 

Les principales fonctions sont les suivantes :

- **Nettoyage des Données** : Fonctions pour supprimer les caractères indésirables, remplir les données manquantes, et assurer la cohérence.
- **Standardisation** : Fonctions pour standardiser les formats des données numériques et catégorielles.
- **Détection et Suppression des Valeurs Aberrantes** : Identification des statistiques de base pour traiter les incohérences potentielles / valeurs aberrantes dans le jeu de données.

### 2. fonctions_cleaning_text.py

Ce fichier se concentre sur le nettoyage des données textuelles pour les applications TALN. 

Il comprend :

- **Suppression des Stopwords** : Suppression des mots courants qui ne contribuent pas au sens du texte.
- **Normalisation du Texte** : Conversion du texte en minuscules, suppression de la ponctuation et expansion des contractions.
- **Stemming/Lemmatisation** : Réduction des mots à leur forme de base ou racine.

### 3. fonctions_analyse_text.py

Ce fichier inclut des fonctions permettant une première analyse des données textuelles. 

Les principales fonctionnalités sont les suivantes :

- **Tokenisation** : Transformation du texte brut en mots ou tokens individuels.
- **Analyse de Fréquence** : Calcul des fréquences des mots dans un texte (tfidf, etc.).
- **Génération de N-grams** : Création de séquences de n-grammes pour capturer le contexte.
- **Analyse de Sentiment** : Analyse du sentiment à l'aide de Vader (si applicable).

## Installation

1. **Cloner le dépôt :**
   ```bash
   git clone https://github.com/mtzleanne/nlp-toolbox.git
   cd repository
   ```

2. **Installer les bibliothèques nécessaires :**
   Assurez-vous que Python est installé. Installez les packages nécessaires à l'aide du fichier d'environnement.

## Exemple d'Utilisation
   
   Importez `fonctions_analyse_text.py` dans votre script Python ou votre notebook Jupyter et appelez les fonctions pertinentes pour analyser vos données textuelles.

   Exemple :
   ```python
   import pandas as pd

   ### Nettoyage d'un document
   import src.toolbox.fonctions_cleaning_text as fonctions_cleaning_text

   fonctions_cleaning_text.preprocess_reponses_notices("Ceci est un exemple de texte")

   ### Analyse d'un document
   import src.toolbox.fonctions_analyse_text as fonctions_analyse_text

   df_docs = pd.DataFrame({"id": [1, 2],
                           "docs" : [["Voici", "premier", "exemple"], 
                           ["autre", "exemple"]]})
   tfidf = fonctions_analyse_text.compute_tf_idf_matrix(df_docs, "docs")

   top3_termes = fonctions_analyse_text.compute_top_n_tf_idf_by_doc(tfidf, 3)
   stats_tf_idf = fonctions_analyse_text.compute_stats_tf_idf_all_docs(tfidf)
   ```
   
Vous pouvez également lancer le script `exemple_utilisation.py` pour le même résultat.

## Contact

Pour toute question ou demande, veuillez contacter [Léanne MARTINEZ](mailto:martinez.leanne.thiers@gmail.com).