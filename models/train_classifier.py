# load libraries
from sqlalchemy import create_engine, text
import ssl

import pandas as pd
import numpy as np

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.tokenize import RegexpTokenizer, word_tokenize

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier

# deactivate ssl certificate to download the nltk package
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('wordnet')


def load_data():
    # create the engine for the sql database
    engine = create_engine('sqlite:///cleaned_data_sql.db')
    
    # create a connection
    conn = engine.connect()
    
    # transform to a executable object for pandas
    sql = text("SELECT * FROM clean_disaster_messages")
    
    # create the dataframe
    df = pd.read_sql(sql, conn)
    
    # data split
    X = df['message']
    y = df.drop(columns=['message'])
    
    # delete number-only rows 
    row_count = -1
    for row in X:
        row_count =+ 1
        if type(row) == int or type(row) == float:
            X.drop(row_count)
    return X, y

# displaying the ml performance
def display_results(y_test, y_pred):
    labels = np.unique(y_pred)
    confusion_mat = confusion_matrix(y_test['related'], y_pred['related'], labels=labels)
    accuracy = (y_pred == y_test).mean()

    print("Labels:", labels)
    print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)
 
# machine learning pipline   
def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier())
    ])

    # train classifier
    pipeline.fit(X_train, y_train)

    # predict on test data
    y_pred = pipeline.predict(X_test)

    # display results
    display_results(y_test, y_pred)