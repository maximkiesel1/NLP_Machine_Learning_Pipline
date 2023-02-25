# NLP_Pipline_disaster_response

This Python package contains an NLP machine learning pipeline that processes and cleans raw data, stores it in a SQL database, retrieves the data from the SQL database, processes it in a machine learning pipeline (using bow, pos, tf-idf, lemmatization), and trains a random forest. The best parameters for the model are found using grid search before the model is trained. The resulting model is then saved in a classifier.pkl file. The package also includes a web app (app.py) that loads the data and transfers it to a web app, allowing other texts to be processed using the machine learning model.

# Directory Structure

The package has the following directory structure:

- NLP_Pipline_disaster_response (Main folder)
  - data (sub-folder)
    - disaster_messages.csv
    - disaster_categories.csv
    - process_data.py
    - cleaned_data_sql.db
  - models (sub-folder)
    - train_classifier.py
    - text_length_extractor.py
    - classifier.pkl
  - app (sub-folder)
    - run.py
    - templates (sub-folder)
      - go.html
      - master.html

# Data

The data folder contains the following files:

- disaster_messages.csv: Contains raw disaster-related text messages.
- disaster_categories.csv: Contains the categories for each disaster-related message.
- process_data.py: A Python script that reads in the disaster_messages.csv and disaster_categories.csv files, cleans the data, and saves the resulting data in a SQLite database called cleaned_data_sql.db.
- cleaned_data_sql.db: A SQLite database containing the cleaned data from the disaster_messages.csv and disaster_categories.csv files.

# Models

The models folder contains the following files:

- train_classifier.py: A Python script that reads in the cleaned data from the cleaned_data_sql.db file, processes it using a machine learning pipeline (using bow, pos, tf-idf, lemmatization), and trains a random forest. The best parameters for the model are found using grid search. The resulting model is then saved in a file called classifier.pkl.
- text_length_extractor.py: A Python script that extracts the length of each text message in characters and words.
- classifier.pkl: A file containing the trained machine learning model.

# Web App

The app folder contains the following files:

- run.py: A Python script that loads the trained machine learning model from the classifier.pkl file and uses it to classify new text messages.
- templates (sub-folder): A sub-folder containing the following files:
- go.html: A web page that displays the classification results for a single text message.
- master.html: A web page that displays the classification results for multiple text messages.

# How to use the package

To use the package, follow these steps:

- Clone the repository to your local machine.
- Install the required packages by running pip install -r requirements.txt in the root directory of the repository.
- Navigate to the data folder and run python process_data.py. This will read in the disaster_messages.csv and disaster_categories.csv files, clean the data, and save the resulting data in a SQLite database called cleaned_data_sql.db.
- Navigate to the models folder and run python train_classifier.py. This will read in the cleaned data from the cleaned_data_sql.db file, process it using a machine learning pipeline (using bow, pos, tf-idf, lemmatization), and train a random forest.
