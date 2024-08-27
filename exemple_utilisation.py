import pandas as pd

### Cleaning du document
import src.toolbox.fonctions_cleaning_text as fonctions_cleaning_text

fonctions_cleaning_text.preprocess_reponses_notices("Ceci est un example de texte")

### Analyse du document
import src.toolbox.fonctions_analyse_text as fonctions_analyse_text

df_docs = pd.DataFrame({"id": [1, 2],
                        "docs" : [["Voice", "premier", "exemple"], 
                        ["autre", "example"]]})
tfidf = fonctions_analyse_text.compute_tf_idf_matrix(df_docs, "docs")

top3_termes = fonctions_analyse_text.compute_top_n_tf_idf_by_doc(tfidf, 3)
stats_tf_idf = fonctions_analyse_text.compute_stats_tf_idf_all_docs(tfidf)

