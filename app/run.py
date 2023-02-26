import json
import plotly
import pandas as pd
import sys

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine, text

import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.tokenize import word_tokenize

from text_length_extractor import TextLengthExtractor


app = Flask(__name__)

def tokenize(message):
    '''
    Cleaning, lower letters, tokenize, remove stopwords and lemmatize on the base of part-of-speech.
    
    INPUT 
    message - single message of the corpus
      
    OUTPUT
    clean_message - single, cleaned message of the corpus
    '''
    # remove url´s from message
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    message = url_pattern.sub(' ', message)
    
    # remove all symboles which are not letter or numbers
    sym_pattern = re.compile(r'[^a-zA-Z0-9]')
    message = sym_pattern.sub(' ', message)
    
    # transfrom to lower case
    message = message.lower()
    
    # tokenize message
    message = word_tokenize(message)
    
    # apply part of speech tagging
    message = pos_tag(message)
    
    # remove stop words
    stop_words = set(stopwords.words("english"))
    message = [word for word in message if word[0] not in stop_words]

    # find the right index for the root word in the WordNetLemmatizer().lemmatize
    def pos_var(word):
        pos_value = 0
        if word[1].startswith('J'):
            pos_value = 'a'
        elif word[1].startswith('V'):
            pos_value = 'v'
        elif word[1].startswith('N'):
            pos_value = 'n'
        elif word[1].startswith('R'):
            pos_value = 'r'
        else:
            pos_value = 'n'
        return pos_value
    
    # lemmatize function for the word in the word-tag-pair
    word_tag_count = -1
    for word_tag in message:
        word_tag_count += 1
        message[word_tag_count] = WordNetLemmatizer().lemmatize(word_tag[0], pos=pos_var(word_tag))
    
    clean_message = message
    
    return clean_message


# load data
engine = create_engine('sqlite:///../data/cleaned_data_sql.db')

# create a connection
conn = engine.connect()
    
# transform to a executable object for pandas
sql = text("SELECT * FROM cleaned_data")
    
# create the dataframe
df = pd.read_sql(sql, conn)

# load model
model = joblib.load("../models/classifier.pkl")


def count_pos_class(df):
    '''
    Find the names and amount of positive target classes in the dataframe
    
    INPUT
    df - dataframe which have one feature and the rest are target columns with 0 or 1 values
    
    OUTPUT
    numb_pos_class - amount of the positive values for each target column
    name_pos_class - names of the columns with positive values
    '''
    numb_pos_class = []
    name_pos_class = []
    for i in df:
        if i == 'message':
            next
        else:
            try:
                numb_pos_class.append(
                    list(df.groupby(i).count()['message'])[1]
                )
                name_pos_class.append(i)
            except:
                next
    return numb_pos_class, name_pos_class

def top_dif_class(df):
    '''
    Find how many different classes have a message in the corpus. Sorte by the lowest to the highest.
    
    INPUT
    df - dataframe which have one feature and the rest are target columns with 0 or 1 values
    
    OUTPUT
    list_count - list with the unique counts of messages with different classes
    top_count - list with names for the unique counts
    '''
    list_count = []
    for index, row in df.iterrows():
        class_count = 0
        for i in row:
            if i == '1':
                class_count += 1
        list_count.append(class_count)
    
    list_count = list(set(list_count))
    
    top_count = []

    for i in range(1, len(list_count)+1):
        top_count.append(f"Top{i}")
    
    return list_count, top_count

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    numb_pos_class, name_pos_class = count_pos_class(df)
    
    list_count, top_count = top_dif_class(df)
    
    # create visuals

    graphs = [
        {
            'data': [
                Bar(
                    x=name_pos_class,
                    y=numb_pos_class
                )
            ],

            'layout': {
                'title': 'Distribution Positive Classes Of Messages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Names With Positive Classes",
                    'tickangle': 35
                }
            }
        },
        {
            'data': [
                Bar(
                    x = top_count,
                    y = list_count
                )
            ],

            'layout': {
                'title': 'Count Of Different Classes In Single Messages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Top Counts",
                    'tickangle': 35
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html',
                           ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()