"""
Author : Léanne MARTINEZ
"""

import numpy as np
import pandas as pd
import re
import os
import pickle
import math

from wordcloud import WordCloud
import matplotlib.pyplot as plt

from collections import Counter
from itertools import chain
from itertools import product

from unidecode import unidecode
import gensim
from gensim import corpora
from gensim.models import CoherenceModel
import nltk
from nltk.util import ngrams
from nltk.probability import FreqDist
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

from vaderSentiment_fr.vaderSentiment import SentimentIntensityAnalyzer


nlp_fr = spacy.load("fr_core_news_sm")

from . import perso_config
from . import fonctions_cleaning

# ---------------------------------------------------------------------------------------------------------------------------------------------
### Définition d'une fonction qui créée un nuage de mots à partir de tokens
# ---------------------------------------------------------------------------------------------------------------------------------------------
def create_wordcloud_from_token(df: pd.DataFrame, col):
    """
    Créer un nuage de mot à partir d'une colonne d'un Dataframe / d'un Dataframe composé.e de tokens

    @param df: Pandas Dataframe
    @param col : str pour le nom de la colonne qui contient les listes de token ou False
    @return : void
    """
    if col != False:
        wc = WordCloud(
            background_color="white", mode="RGBA", max_words=100, width=1600, height=800
        ).generate(df[col].dropna().to_string())
        wc.to_file(
            os.path.join(
                perso_config.output_nuages_mots_path, f"nuage_mot_{col}.png"
            )
        )
    else:
        wc = WordCloud(
            background_color="white", mode="RGBA", max_words=100, width=1600, height=800
        ).generate(df.dropna().to_string())
        wc.to_file(
            os.path.join(
                perso_config.output_nuages_mots_path, f"nuage_mot_all.png"
            )
        )


# ---------------------------------------------------------------------------------------------------------------------------------------------
### Définition d'une fonction qui calcule le TF-IDF à partir de tokens
# ---------------------------------------------------------------------------------------------------------------------------------------------
def compute_tf_idf_matrix(df: pd.DataFrame, col: str):
    """
    Créer la matrice comportant les td-idf de chaque document

    @param df: Pandas Dataframe
    @param col : str pour le nom de la colonne qui contient les listes de token
    @return df_tfidfvect : Matrice de TF-IDF mots x documents
    """
    data_res = df[col].dropna().to_list()

    def dummy_fun(
        doc,
    ):  # Peut-être modifiée et remplacée par preprocess_reponses_notices par exemple
        return doc

    # On initialise le vectorizer tf-idf de scikit-learn
    tfidf_vec = TfidfVectorizer(
        analyzer="word",
        tokenizer=dummy_fun,
        preprocessor=dummy_fun,
        token_pattern=None,
        use_idf=True,
        norm="l2",
    )
    # On le calibre sur la base de données de tokens
    tfidf_vec.fit(data_res)

    # On le lance sur la base de données de tokens
    tfidf = tfidf_vec.fit_transform(data_res)

    # On extrait les résultats et on les enregistre dans un dataframe que l'on retourne
    tfidf_tokens = tfidf_vec.get_feature_names_out()
    df_tfidfvect = pd.DataFrame(data=tfidf.toarray(), columns=tfidf_tokens)
    return df_tfidfvect


# ---------------------------------------------------------------------------------------------------------------------------------------------
### Définition d'une fonction qui calcule des statistiques sur la matrice de TF-IDF
# ---------------------------------------------------------------------------------------------------------------------------------------------
def compute_top_n_tf_idf_by_doc(tf_idf_matrix: pd.DataFrame, n: int):
    """
    Calcule le top n des plus grands tf-idf par documents (1 doc / col) à partir de la matrice TF-IDF mots x documents

    @param tf_idf_matrix : Pandas Dataframe de la matrice TF-DIF mots x documents
    @param n : int pour le nombre de termes qu'on souhaite conserver
    @return df_tfidfvect : Matrice de TF-IDF mots x documents
    """
    ### Top 5 des plus grand tf-idf par documents
    df_tfidfvect_transpose = tf_idf_matrix.T
    largest_tf_idf_by_doc = df_tfidfvect_transpose.apply(
        lambda s: pd.Series(s.nlargest(n).index)
    )
    return largest_tf_idf_by_doc


def compute_stats_tf_idf_all_docs(tf_idf_matrix: pd.DataFrame):
    """
    Calcule des statistique basiques sur les tf-idf à partir de la matrice TF-DIF mots x documents

    @param tf_idf_matrix : Pandas Dataframe de la matrice TF-DIF mots x documents
    @return df_tfidfvect : Matrice de TF-IDF mots x documents
    """
    res = pd.DataFrame(fonctions_cleaning.compute_stats_basics(tf_idf_matrix))
    return res


# ---------------------------------------------------------------------------------------------------------------------------------------------
### Définition d'une fonction qui compte les ngrams à partir de listes de tokens
# ---------------------------------------------------------------------------------------------------------------------------------------------
def count_ngrams(df: pd.DataFrame, col: str, n: int):
    """
    Calcule le volume d'occurrence d'un ngrams dans une colonne de Dataframe composée de listes de tokens

    @param df: Pandas Dataframe
    @param col : str pour le nom de la colonne qui contient les listes de token
    @param n : int pour le nombre de tokens à conserver dans le ngram
    @return df_ngram_counts : Pandas Dataframe avec le compte des ngrams classés par nombre d'occurence
    """
    # On définit une liste de liste de tokens à partir des données (documents)
    df_res_list = df[col].dropna().to_list()

    # On compte les ngrams à l'aide de Counter (package collections)
    ngram_counts = Counter()

    for token_list in df_res_list:
        token_list_ngrams = ngrams(token_list, n)
        ngram_counts.update(token_list_ngrams)

    # On formate le dataframe à retourner
    df_ngram_counts = pd.DataFrame(ngram_counts, index=["count"])
    df_ngram_counts = df_ngram_counts.T.sort_values(
        "count", ascending=False
    ).reset_index()
    df_ngram_counts.columns = [f"token_{i}" for i in range(0, n)] + ["count"]

    # On définit une variable qui contient tous les token concaténés dans un seul str (afin de simplifier les recherches de token précises)
    df_ngram_counts["token_concat"] = df_ngram_counts.apply(
        lambda row: ";".join(map(str, row[[f"token_{i}" for i in range(0, n)]])), axis=1
    )

    # On retourne le dataframe de comptages
    return df_ngram_counts


# ---------------------------------------------------------------------------------------------------------------------------------------------
### Définition d'une fonction qui crée des clusters de documents (K-means)
# ---------------------------------------------------------------------------------------------------------------------------------------------
def attribute_cluster_by_rep_tokenised(
    df: pd.DataFrame, col: str, n: int, input_method="SVD", nb_components = 200
):
    """
    Calcule le volume d'occurrence d'un ngrams dans une colonne de Dataframe composée de listes de tokens

    @param df: Pandas Dataframe
    @param col : str pour le nom de la colonne qui contient les listes de token
    @param n : nombre de clusters pour le k-means
    @param input_method : str pour le type de matrice en entrée du K-means (SVD pour réduire la dimension, ou TF-IDF)
    @param nb_components : nombre de composantes pour la SVD, par defaut à 200
    @return [df_res_q, X_svd ou X_tfidf, labels]: Pandas Dataframe avec les documents tokénisées et leur cluster attribué,
                                        Matrice d'entrée,
                                        Liste d'attribution des clusters
    """
    # On définit les méthodes possibles
    methods = [
        "SVD",  # On base le k-means sur une matrice dont on a réduit la dimension (truncated SVD)
        "TF-IDF",
    ]  # On base le k-means sur une matrice TF-IDF (Vectorizer)
    if input_method not in methods:
        raise ValueError(
            "Argument input_method invalide. Chosir dans la liste suivante : %s"
            % methods
        )

    # On définit une liste de liste de tokens à partir des données (documents)
    data_res = df[col].dropna().to_list()

    # Définir une fonction lambda pour donner à la place des paramètres par défaut
    def dummy_fun(
        doc,
    ):  # Peut-être modifiée et remplacée par preprocess_reponses_notices par exemple
        return doc

    ### K-means avec la truncated SVD
    if input_method == "SVD":
        # On initialise le Countvectorizer pour pouvoir calculer la matrice document-terme
        count_vec = CountVectorizer(
            analyzer="word",
            tokenizer=dummy_fun,
            preprocessor=dummy_fun,
            token_pattern=None,
        )
        # On calcule la matrice documents - termes
        rep_term = count_vec.fit_transform(data_res)

        # On définit puis on applique le modèle SVD sur la matrice documents - termes (équivalent d'une ACP pour des matrices rectanglaires) - permet de réduire la dimension
        svd = TruncatedSVD(
            n_components=nb_components,  # On réduit à 200 composantes, possible d'adapter en fonction de la taille du dictionnaire (nombre de tokens total)
            random_state=42,
        )
        X_svd = svd.fit_transform(rep_term)

        # On affiche la part de variable expliquée
        print(
            f"Part de variance expliquée : {np.sum(svd.explained_variance_ratio_):.2f}"
        )

        ### Normalisation cosine
        i = 0
        sum_sqrt_vals_squared = dict()
        for rep in X_svd:
            j = 0
            vals_squared = dict()
            for val in rep:
                vals_squared[j] = val * val
                j += 1
            sum_sqrt_vals_squared[i] = math.sqrt(sum(vals_squared.values()))
            i += 1

        sum_sqrt_vals_squared = []
        for rep in X_svd:
            j = 0
            vals_squared = dict()
            for val in rep:
                vals_squared[j] = val * val
                j += 1
            sum_sqrt_vals_squared.append(math.sqrt(sum(vals_squared.values())))

        # Definir la formule à appliquer à chaque colonne
        def normalisation(row, normalisation_list):
            return row / normalisation_list

        # On applique la normalisation
        X_svd_normalized = np.apply_along_axis(
            normalisation, axis=0, arr=X_svd, normalisation_list=sum_sqrt_vals_squared
        )
        print(X_svd_normalized)

        # On lance le kmeans
        kmeans_svd = KMeans(
            n_clusters=n,
            max_iter=5000,
            n_init=200,
            init="k-means++",
            random_state=1234,  # Pour assurer la reproducibilité
        ).fit(X_svd_normalized)
        cluster_ids, cluster_sizes = np.unique(kmeans_svd.labels_, return_counts=True)
        print(f"Nombre d'éléments assignés à chaque cluster : {cluster_sizes}")

        # On définit le dataframe à retourner et on y ajoute une colonne qui attribue les clusters par documents
        df_res_q = df[[col]].dropna()
        df_res_q["cluster_SVD"] = kmeans_svd.labels_
        labels = kmeans_svd.labels_

        # On retoune le dataframe avec les docuemnts tokenisés et le cluster attribué
        return [df_res_q, X_svd, X_svd_normalized, svd, labels, count_vec]

    ### K means avec la matrice tf-idf
    elif input_method == "TF-IDF":
        # On initialise le vectorizer tf-idf de scikit-learn
        tfidf_vec = TfidfVectorizer(
            analyzer="word",
            tokenizer=dummy_fun,
            preprocessor=dummy_fun,
            token_pattern=None,
            use_idf=True,
            norm="l2",
        )
        # On le calibre sur la base de données de tokens
        tfidf_vec.fit(data_res)

        # On le lance sur la liste de liste de tokens
        X_tfidf = tfidf_vec.fit_transform(data_res)
        print(
            f"Nombre de documents : {X_tfidf.shape[0]}, Nombre de tokens : {X_tfidf.shape[1]}"
        )

        # On lance le kmeans
        kmeans_tfidf = KMeans(
            n_clusters=n,
            max_iter=5000,
            n_init=200,
            init="k-means++",
            random_state=1234,  # Pour assurer la reproducibilité
        ).fit(X_tfidf)
        cluster_ids, cluster_sizes = np.unique(kmeans_tfidf.labels_, return_counts=True)
        print(f"Nombre d'éléments assignés à chaque cluster : {cluster_sizes}")

        # On définit le dataframe à retourner et on y ajoute une colonne qui attribue les clusters par documents
        df_res_q = df[[col]].dropna()
        df_res_q["cluster_matrix_tfidf"] = kmeans_tfidf.labels_

        # On retoune le dataframe avec les documents tokenisée et le cluster attribué
        return [df_res_q, X_tfidf, labels]


def describe_cluster_reponses(df: pd.DataFrame, subset: str):
    """
    Decrit les clusters de documents à l'aide du TF-IDF moyen

    @param df: Pandas Dataframe issus de la fonction attribute_cluster_by_rep_tokenised
    @param subset: str pour indiquer le nom du subset sur lequel est performé le clustering (pour le nom de l'output)
    @return df_res_q : Pandas Dataframe avec les documents tokénisées et leur cluster attribué
    """
    ### On calcule combien de clusters ont été définis
    question_col = df.columns[df.columns.str.endswith("token")][0]
    question_col = "_".join(question_col.replace("_token", "").split())

    # On tope la colonne qui définit le cluster
    cluster_col_name = df.columns[df.columns.str.startswith("cluster")][0]
    nb_clusters = df[cluster_col_name].nunique()

    # On initialise le dictionnaire dans lequel on va enregistrer les résultats
    dict_desc_clust = dict()

    # On boucle sur le nombre de clusters
    for clust_id in range(0, nb_clusters):
        # On définit un sous dataframe qui ne contient que les documents tokénisées du cluster clust_id
        df_cluster = df.loc[df[cluster_col_name] == clust_id]

        # On calcule la matrice de tfidf sur ces documents-ci
        df_cluster = compute_tf_idf_matrix(df_cluster, df_cluster.columns[0])

        # On calcule la moyenne et les quantiles
        df_cluster_stats = compute_stats_tf_idf_all_docs(df_cluster)

        # On trie le dataframe
        df_cluster_stats = df_cluster_stats.sort_values("Moyenne", ascending=False)

        print(
            "Cluster ",
            clust_id,
            "Nombre de documents concernées :",
            df_cluster.shape[0],
            df_cluster_stats.head(10),
        )

        # On incrémente le dictionnaire de résultats
        dict_desc_clust[f"{clust_id} - {df_cluster.shape[0]}"] = df_cluster_stats

    with pd.ExcelWriter(
        os.path.join(
            perso_config.output_stats_desc_desc_cluster_folder,
            f"desc_clusters_{question_col}_{nb_clusters}_{subset}.xlsx",
        )
    ) as writer:
        # Iterate over the dictionary items and write each DataFrame to a separate sheet
        for sheet_name, df in dict_desc_clust.items():
            df.reset_index().to_excel(
                writer, sheet_name=f"Cluster {sheet_name}", index=False
            )

    # On retourne le dictionnaire de résultats complet
    return dict_desc_clust


# ---------------------------------------------------------------------------------------------------------------------------------------------
### Définition d'une fonction qui performe une NDA (identification de sujets dans un corpus)
# ---------------------------------------------------------------------------------------------------------------------------------------------
def dectect_topics_LDA(
    df: pd.DataFrame, col=str, nb_topics=int, nb_passes=int, chunk_size=int
):
    """
    Calcule les sujets d'un corpus avec la méthode Latente Dirichlet Analysis

    @param df : Le dataframe qui contient les listes de tokens par questions
    @param col : La colonne de la question pour laquelle on veut extraire les thèmes
    @param nb_topics : nombre de sujets
    @param nb_passes : nombre de passes sur les documents (hyper-paramètre de l'optimisation)
    @param chunk_size : taille des mini-batchs (hyper-paramètre de l'optimisation)
    @return df_topic_model : Dataframe avec les mots les plus importants par sujets (sujets en lignes, mots en colonnes)
    """
    # Formater les données
    df_res_list = df[col].dropna().to_list()
    dictionary = gensim.corpora.Dictionary(df_res_list)
    dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n=1000)
    bow_corpus = [dictionary.doc2bow(doc) for doc in df_res_list]

    # Créer et lancer le modèle
    lda_model = gensim.models.LdaModel(
        bow_corpus,
        num_topics=nb_topics,
        id2word=dictionary,
        passes=nb_passes,
        chunksize=chunk_size,
    )
    # Afficher les topics
    topics = []
    for idx, topic in lda_model.print_topics(-1):
        print("Topic: {} -> Words: {}".format(idx, topic))
        topics.append(topic)

    # Checker la cohérence du modèle (censé être entre 0.4 et 0.7)
    coherence_model_lda = CoherenceModel(
        model=lda_model, texts=df_res_list, dictionary=dictionary
    )
    coherence_lda = coherence_model_lda.get_coherence()
    print("Coherence Score: ", coherence_lda)

    # Créer la liste des sorties
    all_topic_model = []
    for i in range(len(topics)):
        str = topics[i].split(" + ")
        topic_model = []
        for j in range(10):
            weight = str[j][0:5]
            word = str[j][7 : len(str[j]) - 1]
            topic_model.append((weight, word))
        all_topic_model.append(topic_model)

    # Traduction en DataFrame et renommage
    df_topic_model = pd.DataFrame(all_topic_model)
    mapping = {i: f"Topic {i + 1}" for i in range(nb_topics)}
    df_topic_model.rename(index=mapping, inplace=True)

    return df_topic_model


# ---------------------------------------------------------------------------------------------------------------------------------------------
### Définition d'une fonction qui calcule si une document mentionne un sujet choisi (à partir d'un dico de tuples ou liste de token)
# ---------------------------------------------------------------------------------------------------------------------------------------------
def count_rep_mentionning_chosen_topic(
    df: pd.DataFrame, col: str, search_topic_obj, taille_ngrammes: int, topic: str
):
    """
    Calcule les sujets d'un corpus avec la méthode Latente Dirichlet Analysis

    @param df : Le dataframe qui contient les listes de tokens par questions
    @param col : La colonne de la question pour laquelle on veut extraire les thèmes
    @param search_topic_obj : liste/dict des token/tuples qui correspondent au sujet
    @param taille_ngrammes : taille des n-grammes dans lesquels on cherche les mots
    @param topic : str pour décrire le topic
    @return void : Ajoute une colonne de booléen au dataframe en entrée pour flager si l'axe est mentionné
    """
    df[f"flag_topic_{topic}_{col}"] = False

    # On cherche des tuples
    if type(search_topic_obj) == dict:
        # On calcule toutes les combinaisons de tokens recherchés à partir du dictionnaire
        search_topic_list_combin = list(product(*search_topic_obj.values()))

        # On boucle sur les lignes de la colonne qui nous intéresse
        for token_list in df[col]:
            # Condition pour écarter les NA
            if token_list == token_list:

                all_found_n_grammes = []

                # On boucle sur les combinaisons
                for searched_combin in search_topic_list_combin:

                    # On initialise une liste pour stocker les tokens du df qui commencent par les tokens de recherche
                    found_tokens = []

                    # On boucle sur les tokens de cette liste
                    for token in token_list:
                        # Toper si les tokens du document commencent comme les tokens recherchés
                        if any(
                            token.startswith(search_token)
                            for search_token in searched_combin
                        ):
                            found_tokens.append(token)

                    found_tokens = set(found_tokens)
                    if len(found_tokens) > 1:

                        # On initialise une liste pour stocker les ngrammes correspondants
                        found_n_grammes = []
                        n_grammes = list(ngrams(token_list, taille_ngrammes))

                        # On boucle sur les n-grammes
                        for gram in n_grammes:
                            # Toper si tous les tokens recherchés sont dans les n-grammes
                            if all(token in gram for token in found_tokens):
                                found_n_grammes.append(gram)

                        all_found_n_grammes.append(found_n_grammes)

                if list(chain.from_iterable(all_found_n_grammes)) != []:
                    # print("ngrammes", topic, ":", all_found_n_grammes)
                    df.loc[
                        df[col].astype(str) == str(token_list),
                        f"flag_topic_{topic}_{col}",
                    ] = True
        print(
            f"Nombre de documents concernées par le sujet {topic} dans la colonne {col}:",
            df[f"flag_topic_{topic}_{col}"].sum(),
        )
    # On ne cherche pas de tuples
    elif type(search_topic_obj) == list:
        # On boucle sur les lignes de la colonne qui nous intéresse
        for token_list in df[col]:
            # Condition pour écarter les NA
            if token_list == token_list:

                all_found_tokens = []

                # On boucle sur les combinaisons
                for searched_token in search_topic_obj:

                    # On initialise une liste pour stocker les tokens du df qui commencent par les tokens de recherche
                    found_tokens = []

                    # On boucle sur les tokens de cette liste
                    for token in token_list:
                        # Toper si les tokens du document commencent comme les tokens recherchés
                        if token.startswith(searched_token):
                            found_tokens.append(token)
                    all_found_tokens.append(found_tokens)

                if list(chain.from_iterable(all_found_tokens)) != []:
                    df.loc[
                        df[col].astype(str) == str(token_list),
                        f"flag_topic_{topic}_{col}",
                    ] = True
        print(
            f"Nombre de documents concernées par le sujet {topic} dans la colonne {col}:",
            df[f"flag_topic_{topic}_{col}"].sum(),
        )

    else:
        print("Erreur dans le type de ", search_topic_obj)


# Define the function to check the condition
def at_least_two_match(gram, found_tokens):
    # Generator expression to count matches
    matches = sum(token in gram for token in found_tokens)
    # Return True if at least two matches, False otherwise
    return matches >= 2

# ---------------------------------------------------------------------------------------------------------------------------------------------
### Définition d'une fonction qui identifie et modifie des ngramms dans des chaines de caractères
# ---------------------------------------------------------------------------------------------------------------------------------------------
def flag_phraser_ngrams(
    df: pd.DataFrame,
    col: str,
    search_topic_obj,
    taille_ngrammes: int,
    replacement: str,
    topic: str,
):
    """
    Calcule les sujets d'un corpus avec la méthode Latente Dirichlet Analysis

    @param df : Le dataframe qui contient les listes de tokens par questions
    @param col : La colonne de la question pour laquelle on veut changer les expressions
    @param search_topic_obj : liste/dict des token/tuples qui correspondent au sujet
    @param taille_ngrammes : taille des n-grammes dans lesquels on cherche les mots
    @param replacement : str pour remplacer le ngramme trouvé
    @topic : str qui décrit le flag phraser
    @return void : Ajoute une colonne de booléen au dataframe en entrée pour flager si l'axe est mentionné
    """
    df[f"response_flag_added_{topic}"] = df[col]

    # On cherche des tuples
    if type(search_topic_obj) == dict:
        # On calcule toutes les combinaisons de tokens recherchés à partir du dictionnaire
        search_topic_list_combin = list(product(*search_topic_obj.values()))

        # On boucle sur les lignes de la colonne qui nous intéresse
        for reponse in df[col]:
            # Condition pour écarter les NA
            if reponse == reponse:

                all_found_n_grammes = []

                # On boucle sur les combinaisons
                for searched_combin in search_topic_list_combin:
                    # On initialise une liste pour stocker les tokens du df qui commencent par les tokens de recherche
                    found_tokens = []

                    # On boucle sur les tokens de cette liste
                    for token in reponse.lower().split():
                        # Toper si les tokens du document commencent comme les tokens recherchés
                        if any(
                            token.startswith(search_token)
                            for search_token in searched_combin
                        ):
                            found_tokens.append(token)

                    found_tokens = set(found_tokens)

                    if len(found_tokens) > 1:

                        # On initialise une liste pour stocker les ngrammes correspondants
                        found_n_grammes = []
                        n_grammes = list(
                            ngrams(reponse.lower().split(), taille_ngrammes)
                        )

                        # On boucle sur les n-grammes
                        for gram in n_grammes:
                            # Toper si tous les tokens recherchés sont dans les n-grammes
                            # if all(token in gram for token in found_tokens):
                            if at_least_two_match(gram, found_tokens):
                                found_n_grammes.append(gram)

                        all_found_n_grammes.append(found_n_grammes)
                        print(all_found_n_grammes)

                if list(chain.from_iterable(all_found_n_grammes)) != []:
                    reponse_new = reponse
                    for found_n_grammes in all_found_n_grammes:
                        if found_n_grammes != []:
                            reponse_new = reponse_new.lower().replace(
                                " ".join(found_n_grammes[0]), replacement
                            )
                    df.loc[
                        df[col].astype(str) == str(reponse),
                        f"response_flag_added_{topic}",
                    ] = reponse_new

# ---------------------------------------------------------------------------------------------------------------------------------------------
### Définition d'une calcule le sentiment à partir du lexicon Vader
# ---------------------------------------------------------------------------------------------------------------------------------------------
### Travail sur le lexicon
# Création de la variable sentiment qui contient le lexicon
sentiment = SentimentIntensityAnalyzer()

# Ajouts de mots
mots_complementaires = {
    ### EXEMPLE
    # "ralentissement": -1,
    # "maitrisé": 0.8,
}
sentiment.lexicon.update(mots_complementaires)

# Retraits de mots
### EXEMPLE
# sentiment.lexicon.pop("crise")  # Sens particulier dans le contexte

def get_sentiment_scores(text):
    """
    Calcule le score composé et la part de mots polarisés dans un texte

    @param text : Le dataframe qui contient les listes de tokens par questions
    @param return : Le score composé (senti["compound"]) et la part de mots polarisés dans un texte (pos, neu, neg)
    """
    senti = sentiment.polarity_scores(text)
    return senti["pos"], senti["neu"], senti["neg"], senti["compound"]
