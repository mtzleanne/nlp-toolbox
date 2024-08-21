# README

## Overview

This repository contains Python scripts designed for text analysis and data cleaning in the context of Natural Language Processing (NLP) tasks. 

The scripts are organized into three main files:

1. **fonctions_cleaning.py**: This script contains functions aimed at cleaning data. It focuses on preprocessing tasks such as removing special characters, handling missing values, and standardizing formats across datasets.

2. **fonctions_cleaning_text.py**: This script extends the functionality of the `fonctions_cleaning.py` by including additional text-specific cleaning functions, like stopword removal, text normalization, and stemming/lemmatization.

3. **fonctions_analyse_text.py**: This script provides functions for analyzing text data, including tokenization, frequency analysis, and other common NLP operations.

## File Descriptions

### 1. fonctions_cleaning.py

This file is dedicated to cleaning raw data to prepare it for analysis. Major functions include:

- **Data Cleaning**: Functions to remove unwanted characters, fill missing data, and ensure consistency.
- **Standardization**: Functions to standardize numeric and categorical data formats.
- **Outlier Detection and Removal**: Identifying basic statistics to be able to deal with potentiel inconsistencies / outliers in the dataset.

### 3. fonctions_cleaning_text.py

This file focuses on cleaning text data for NLP applications. It includes:

- **Stopword Removal**: Removing common words that do not contribute to the meaning of the text.
- **Text Normalization**: Converting text to lowercase, removing punctuation, and expanding contractions.
- **Stemming/Lemmatization**: Reducing words to their base or root form.

### 3. fonctions_analyse_text.py

This file includes functions that help in the analysis of textual data. Key functionalities include:

- **Tokenization**: Splitting text into individual words or tokens.
- **Frequency Analysis**: Calculating word frequencies within a text.
- **N-grams Generation**: Creating sequences of N words to capture context.
- **Sentiment Analysis**: Analyzing the sentiment of the text (if applicable).

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mtzleanne/nlp-toolbox.git
   cd repository
   ```

2. **Install required libraries:**
   Ensure you have Python installed. Install necessary packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

## Usage example

   Import the `fonctions_analyse_text.py` in your Python script or Jupyter notebook and call the relevant functions for analyzing your text data.

   Example:
   ```python
   import pandas as pd
   import toolbox.fonctions_analyse_text as fonctions_analyse_text

   df_docs = pd.DataFrame({"id": [1, 2],
                           "docs" : [["this", "is", "first", "example"], 
                           ["this", "is", "another", "example"]]})
   tfidf = fonctions_analyse_text.compute_tf_idf_matrix(df_docs, "docs")
   ```

## Contact

For any questions or inquiries, please contact [Léanne MARTINEZ](mailto:martinez.leanne.thiers@gmail.com).