import pandas as pd
import toolbox.fonctions_analyse_text as fonctions_analyse_text

df_docs = pd.DataFrame({"id": [1, 2],
                        "docs" : [["this", "is", "first", "example"], 
                        ["this", "is", "another", "example"]]})
tfidf = fonctions_analyse_text.compute_tf_idf_matrix(df_docs, "docs")