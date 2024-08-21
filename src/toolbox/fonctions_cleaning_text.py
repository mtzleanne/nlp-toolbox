"""
Author : Léanne MARTINEZ
"""

import numpy as np
import pandas as pd
import re
import os
import pickle

from wordcloud import WordCloud
import matplotlib.pyplot as plt

from autocorrect import Speller

spell = Speller(lang="fr")
from unidecode import unidecode
import gensim
from gensim import corpora
import nltk

nltk.download("stopwords")
from nltk.probability import FreqDist
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

nlp_fr = spacy.load("fr_core_news_sm")

from . import bdd_config
from . import fonctions_cleaning

# ---------------------------------------------------------------------------------------------------------------------------------------------
### Définition des sets / patterns utiles
# ---------------------------------------------------------------------------------------------------------------------------------------------
### Définition d'un set de stopwords (nltk + customisé)
stopwords = list(nltk.corpus.stopwords.words("french"))
newStopWords = [
    "a",
    "de",
    "et",
    "en",
    "par",
    "l'",
    "etc",
    "etre",
    "avoir",
    "ete",
    "être",
    "eter",
    "sous",
    "sou",
    "aller",
]
stopwords += newStopWords
stopwords = set(stopwords)

### Définition d'un set de ponctuation (customisé)
ponctuation = {
    ",",
    ";",
    ".",
    "\n",
    "!",
    "?",
    "(",
    ")",
    ":",
    "'",
    '"',
    "-",
    ">",
    "<",
    "*",
    "\\",
}
# Création d'un pattern regex pour matcher la ponctuation
pattern_ponctuation = re.compile("|".join(re.escape(p) for p in ponctuation))
pattern_ponctuation_all = re.compile(
    "^(" + "|".join(re.escape(p) for p in ponctuation) + ")+$"
)


# ---------------------------------------------------------------------------------------------------------------------------------------------
### Définition d'une fonction qui normalise la chaine de caractères
# ---------------------------------------------------------------------------------------------------------------------------------------------
def preprocess_normalizer(doc: str):
    """
    Cette fonction a pour but d'appliquer les étapes de normalisation utile sur des données textuelles.

    @param doc: str
    @return: str normalisé
    """
    # 1 - Low case - gestion des majuscules
    output_doc = doc.lower()

    # 2 - Retrait des espaces multiples
    output_doc = re.sub(r"\s{1,}", " ", output_doc)

    # 3 - Unicode - gestion des accents
    output_doc = unidecode(output_doc)

    # 4 - Flag phraser - gestion des expressions
    for expr in bdd_config.dict_expressions.keys():
        pattern = r"\b" + expr + r"\w*"
        output_doc = re.sub(pattern, bdd_config.dict_expressions[expr], output_doc)

    # Retourne une liste de tokens à partir du doc d'entrée
    return output_doc


# ---------------------------------------------------------------------------------------------------------------------------------------------
### Définition d'une fonction qui preprocess la chaine de caractères
# ---------------------------------------------------------------------------------------------------------------------------------------------
def preprocess_reponses_notices(doc: str):
    """
    Cette fonction a pour but d'appliquer les étapes de préprocessing utile avant de mener des analyses sur des données textuelles.
    Cette fonction fait appel à la fonction preprocess_normalizer() définie ci-dessus.

    @param doc: str
    @return output_tokens: liste de tokens
    """

    output_doc = doc

    # 1 - Cleaning des fautes d'orthographe sur l'ensemble d'une colonne avec autocorrect
    # On flage les acronymes que l'ont ne veut pas corriger
    for acronyme in bdd_config.list_acronymes_a_conserver:
        if acronyme in doc:
            output_doc = output_doc.replace(acronyme, f"flagconserve{acronyme}")
    # On lance la fonction de correction
    output_doc = spell(output_doc)
    # On retirer le flag
    output_doc = output_doc.replace("flagconserve", "")

    # 2 - Normalisation
    output_doc = preprocess_normalizer(output_doc)

    # 3 - Mise sous la forme de tokens
    output_doc = nlp_fr(output_doc)
    output_tokens = [token.lemma_ for token in output_doc]
    # [token.lemma_ if token.pos_ == "VERB" else token.text for token in output_doc]

    # 4 - Retrait de la ponctuation (retrait des tokens ponctuations, et de la ponctuation contenue dans un token)
    output_tokens = [
        re.sub(pattern_ponctuation, "", token)
        for token in output_tokens
        if not pattern_ponctuation_all.match(token)
    ]

    # 5 - Retrait des stopwords
    output_tokens = [token for token in output_tokens if token not in stopwords]

    # Retourne une liste de tokens à partir du doc d'entrée
    return output_tokens


# ---------------------------------------------------------------------------------------------------------------------------------------------
### Définition d'une fonction corrige les fautes dans une réponse à l'aide du package Spell
# ---------------------------------------------------------------------------------------------------------------------------------------------
def corriger_fautes(reponse):
    """
    Cette fonction a pour but de corriger une réponse str

    @param reponse: str
    @return reponse_corrigee : str corrigé
    """
    if isinstance(reponse, str):
        reponse_corrigee = spell(reponse)
    else:  # Cas de valuer manquante
        reponse_corrigee = np.nan
    return reponse_corrigee


# ---------------------------------------------------------------------------------------------------------------------------------------------
### Définition d'une fonction qui transforme une liste en str séparés de ; pour faciliter les recherches
# ---------------------------------------------------------------------------------------------------------------------------------------------
def flatten_list_to_string(lst):
    """
    Cette fonction a pour but de transformer une liste en str séparés de ; pour faciliter les recherches à l'aide de str.contains

    @param doc: liste
    @return : string
    """
    if type(lst) == type([]):
        return ";".join(map(str, lst))
    else:
        return ""


# ---------------------------------------------------------------------------------------------------------------------------------------------
### Définition d'une fonction qui concatène des listes
# ---------------------------------------------------------------------------------------------------------------------------------------------
def concat_lists(row):
    concatenated_list = []
    for val in row:
        if isinstance(val, list):  # Pour exclure les NA
            concatenated_list.extend(val)
    return concatenated_list


# ---------------------------------------------------------------------------------------------------------------------------------------------
### Définition d'une fonction qui concatène des réponses dans une seule chaine de caractères
# ---------------------------------------------------------------------------------------------------------------------------------------------
def concat_reps(row):
    concatenated_reps = ""
    for val in row:
        if isinstance(val, str):  # Pour exclure les NA
            concatenated_reps += " ; " + val
    return concatenated_reps


# ---------------------------------------------------------------------------------------------------------------------------------------------
### Définition d'une fonction lambda pour donner à la place des paramètres par défaut
# ---------------------------------------------------------------------------------------------------------------------------------------------
def dummy_fun(
    doc,
):
    return doc
