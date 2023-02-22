# load libraries
import sys

from sqlalchemy import create_engine, text
import ssl

import pandas as pd
import pickle

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.tokenize import word_tokenize

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report

from text_length_extractor import TextLengthExtractor


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

def load_data(database_filepath):
    '''
    Loading the data from a sql database and transform it to X, y data for the machine learning model.
    
    INPUT 
    database_filepath - path to the sql database
      
    OUTPUT
    X - Features
    y - Target variables
    '''
    # create the engine for the sql database
    engine = create_engine(f'sqlite:///{database_filepath}')
    
    # create a connection
    conn = engine.connect()
    
    # transform to a executable object for pandas
    sql = text("SELECT * FROM cleaned_data")
    
    # create the dataframe
    df = pd.read_sql(sql, conn)
    
    # data split
    X = df['message']
    y = df.drop(columns=['message'])
    y = y.astype('int')
    
    # delete number-only rows 
    row_count = -1
    for row in X:
        row_count =+ 1
        if type(row) == int or type(row) == float:
            X.drop(row_count)
    return X, y


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


def build_model():
    '''
    Machine learning pipelinie for an NLP use case which use Count-Vectorizer, TF-IDF and 
    Text Length Extracting. After that it finds with Gridsearch the best
    parameters for a RandomForest Classifier.
    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('txt_len', TextLengthExtractor())
        ])),

        ('clf', RandomForestClassifier())
    ])
    
    # define the split for gridsearch
    shuffle = ShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
    
    # define set of hyperparameters for test of the ml model with gridsearch

    parameters = {
        'clf__max_depth': [400, 600],
        'clf__n_estimators': [800, 1000]
    }
    
    model = GridSearchCV(pipeline, param_grid=parameters, cv=shuffle)

    return model



def evaluate_model(model, X_test, y_test):
    '''
    Cleaning, lower letters, tokenize, remove stopwords and lemmatize on the base of part-of-speech.
    
    INPUT 
    model - machine Learning Pipeline in a NLP Use Case
    X_test - splitted test features 
    y_test - splitted test, target variables
      
    OUTPUT
    None
    '''
    target_names = list(y_test.columns)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=target_names)
    print("Labels:", target_names)
    print("Classification Report:\n", report)
    print("\nBest Parameters:", model.best_params_)


def save_model(model, model_filepath):
    '''
    Transform and save the trained model in a pickle file.
    
    INPUT 
    model - machine Learning Pipeline in a NLP Use Case
    model_filepath - path to the file
    
    OUTPUT
    None
    '''
    # save the model as a pickle file
    with open(model_filepath, 'wb') as model:
        pickle.dump(model, model)


def main():
    '''
    Main function to start the pipeline process.
    '''
    
    if len(sys.argv) == 3: # Check if there are 3 inputs
        
        # assign the variables 
        database_filepath, model_filepath = sys.argv[1:]
       
        # print a loading status
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        
        # assign the X and y variables
        X, y = load_data(database_filepath)
        
        # splitting in train and test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        
        # print a building status
        print('Building model...')
        
        # definie model
        model = build_model()
        
        # print a training status
        print('Training model...')
        
        # train the model
        model.fit(X_train, y_train)
        
        # print a evaluating status
        print('Evaluating model...')
        
        # evaluate the model
        evaluate_model(model, X_test, y_test)

        # print a saving status
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        
        # save the model as a pickle file
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()