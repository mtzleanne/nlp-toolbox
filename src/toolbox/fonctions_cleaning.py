"""
Author : Léanne MARTINEZ
"""

import numpy as np
import pandas as pd

import re
from unidecode import unidecode

import os

from . import bdd_config

# ---------------------------------------------------------------------------------------------------------------------------------------------
### Définition d'une fonction qui transforme les faux-NA en vrai NA
# ---------------------------------------------------------------------------------------------------------------------------------------------
def clean_NA_quali(df: pd.DataFrame, dict_rep_defaut: dict):
    """
    Nettoie les faux-NA pour les transformer en np.NaN dans une base composée de données qualitatives uniquement
    @param df: la table pivotée que l'on souhaite nettoyer
    @return: la table même avec les NA nettoyée
    """

    df_clean = df.copy()

    for col in list(df_clean.columns):
        print(col)
        # Mise au format charactère (seulement si nous n'avons pas de données quantitatives dans la base)
        df_clean[col] = [
            row.replace("…", "...").replace("’", "'") if row == str(row) else row
            for row in df_clean[col]
        ]

        # Retrait des espaces inutiles
        df_clean[col] = [
            re.sub(r"\s{1,}", " ", row).strip() if row == str(row) else row
            for row in df_clean[col]
        ]

        # Retrait des réponses par défaut
        df_clean[col] = [
            row.replace("(Zone de saisie)", "") if row == str(row) else row
            for row in df_clean[col]
        ]
        for nom_col in list(dict_rep_defaut.keys()):
            if col == nom_col:
                df_clean[col] = [
                    (
                        row.replace(dict_rep_defaut[nom_col], "")
                        if row == str(row)
                        else row
                    )
                    for row in df_clean[col]
                ]

        # Retrait des réponses de moins de 2 charactères
        df_clean[col] = [
            row if (len(str(row)) > 2) else np.nan for row in df_clean[col]
        ]

    # Retourne le df propre
    return df_clean


# ---------------------------------------------------------------------------------------------------------------------------------------------
### Fonctions pour générer une table récaptitulative du nombre de caractères et de mots par colonnes
# ---------------------------------------------------------------------------------------------------------------------------------------------
def calculate_char_word_distribution(text):
    """
    Calcule le nombre de caractères et de mots dans un str

    @param text: Contenu d'une cellule au format str
    @return num_characters, num_words: Nombre de caractères dans text, Nombre de mots dans text
    """
    if text == str(text):  # Sauter les cas de NA
        num_characters = len(text)
        num_words = len(re.findall(r"\w+", text))
    else:  # Attribuer aux cas de NA 0 caractères et 0 mots
        num_characters = 0
        num_words = 0
    return num_characters, num_words


def generate_summary_table_char_words(
    dataframe: pd.DataFrame, with_na: bool
):
    """
    Génère une table récapitualive du nombre de caractères et de mots pour chaque colonnes d'un dataframe

    @param dataframe: DataFrame Pandas pour lequel on veut calculer la répartition de caractères et de mots
    @return summary_df: DataFrame Pandas récapitulatif
    """
    # On définit la structure du dictionnaire qui va définie la table en sortie
    summary_data = {
        f"colonne": [],
        f"Min Caractères": [],
        f"25% Caractères": [],
        f"50% Caractères": [],
        f"75% Caractères": [],
        f"90% Caractères": [],
        f"Max Caractères": [],
        f"Min Mots": [],
        f"25% Mots": [],
        f"50% Mots": [],
        f"75% Mots": [],
        f"90% Mots": [],
        f"Max Mots": [],
    }

    for column in dataframe.columns:
        print(column)
        # On applique la fonction calculate_char_word_distribution colonne par colonne
        if dataframe[column].dropna().empty == False:
            if with_na:
                char_word_distribution = dataframe[column].apply(
                    calculate_char_word_distribution
                )
            else:
                char_word_distribution = (
                    dataframe[column].dropna().apply(calculate_char_word_distribution)
                )
            num_characters, num_words = zip(*char_word_distribution)

            # On peuple le dictionnaire avec les informations colonne par colonne
            summary_data[f"colonne"].append(column)
            summary_data[f"Min Caractères"].append(np.min(num_characters))
            summary_data[f"25% Caractères"].append(
                np.percentile(num_characters, 25)
            )
            summary_data[f"50% Caractères"].append(
                np.percentile(num_characters, 50)
            )
            summary_data[f"75% Caractères"].append(
                np.percentile(num_characters, 75)
            )
            summary_data[f"90% Caractères"].append(
                np.percentile(num_characters, 90)
            )
            summary_data[f"Max Caractères"].append(np.max(num_characters))

            summary_data[f"Min Mots"].append(np.min(num_words))
            summary_data[f"25% Mots"].append(np.percentile(num_words, 25))
            summary_data[f"50% Mots"].append(np.percentile(num_words, 50))
            summary_data[f"75% Mots"].append(np.percentile(num_words, 75))
            summary_data[f"90% Mots"].append(np.percentile(num_words, 90))
            summary_data[f"Max Mots"].append(np.max(num_words))

    # On transforme le dictionnaire en DataFrame Pandas
    summary_df = pd.DataFrame(summary_data)

    # On retourne le dataframe récapitulatif
    return summary_df

# ---------------------------------------------------------------------------------------------------------------------------------------------
### Fonctions pour générer une table récaptitulative de statistiques basiques
# ---------------------------------------------------------------------------------------------------------------------------------------------
def compute_stats_basics(df: pd.DataFrame):
    """
    Calcule le minimum, P25, P50 (médiane), P75, P90 et maximum d'un dataframe.

    @param df : Le dataframe pour lequel les statistiques doivent être calculées.

    @return dico_res : Un dictionnaire contenant les statistiques calculées.
    """
    dico_res = {
        "Moyenne": df.mean(),
        "Minimum": df.min(),
        "P25": df.quantile(0.25),
        "P50": df.median(),
        "P75": df.quantile(0.75),
        "P90": df.quantile(0.90),
        "Maximum": df.max(),
    }

    return dico_res


def compute_proportion(df: pd.DataFrame, col: str, suffix=""):
    """
    Calcule le pourcentage d'un modalité (NA inclus)

    @param df : Le dataframe pour lequel les statistiques doivent être calculées.
    @param col : La colonne de la variable dont on veut calculer la proportion.
    @param suffix : Indication pour qualifier l'ensemble sur lequel on a calculé les proportions.

    @return dico_res : Un dictionnaire contenant les statistiques calculées.
    """
    # Count the number of occurrences of each value in the series
    value_counts = df[col].value_counts(dropna=False)

    # Calculate the total number of values in the series, including NA values
    total_values = len(df[col])

    # Calculate the proportion of each value in the series
    proportions = round(value_counts / total_values, ndigits=4) * 100
    proportions = pd.DataFrame(proportions).reset_index()

    proportions.rename(
        columns={"count": "_".join(suffix.lower().split())}, inplace=True
    )

    return proportions


# ---------------------------------------------------------------------------------------------------------------------------------------------
### Fonctions pour diviser une colonne par une autre en évitant les divisions par 0
# ---------------------------------------------------------------------------------------------------------------------------------------------
def divide_col(col1, col2):
    return col1 / col2 if col2 != 0 else np.nan
