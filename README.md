# NLP Pipline Disaster Response: A NLP Machine Learning Pipeline Embedded In A Web App
![usgs-k7WetNdaY6A-unsplash](https://user-images.githubusercontent.com/119667336/221402218-b0df9bfe-09a0-4a60-a681-5a239ed89e5c.jpg)


This Python project contains an NLP machine learning pipeline that processes and cleans raw data, stores it in a SQL database, retrieves the data from the SQL database, processes it in a machine learning pipeline (using bow, pos, tf-idf, lemmatization), and trains a random forest. The best parameters for the model are found using grid search before the model is trained. The resulting model is then saved in a classifier.pkl file. The package also includes a web app (app.py) that loads the data and transfers it to a web app, allowing other texts to be processed using the machine learning model.

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
    - text_length_extractor.py
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
  - With this model the following score is reached:
    - Recall = 0.85
    - F1-Score = 0.60
- text_length_extractor.py: A Python script that extracts the length of each text message in characters and words.
- classifier.pkl.zip: A file containing the trained machine learning model.
  - To download the classifier.py.zip file, which has been uploaded to a Git LFS server, you will need to have Git LFS installed and configured. After installing Git LFS, you can activate it by running the command git lfs install. Once Git LFS is installed and configured, you can download the classifier.py.zip file by running git lfs pull. After downloading, the file will be in its compressed zip format and must be manually unzipped by the user.

# Web App

The app folder contains the following files:

- run.py: Start a Flask web application to use the machine learning model in the web app and show some graphs about the training data. 
- text_length_extractor.py: A Python script that extracts the length of each text message in characters and words.
- templates (sub-folder): A sub-folder containing the following files:
  - go.html: Template file for the output page of the web app
  - master.html: Template file for building the overall structure and layout of the web app

# How to use the program

To use the package, follow these steps:

- Clone the repository to your local machine.
- To successfully run this project, the following libraries are required:

  - json
  - plotly
  - pandas
  - nltk
  - Flask
  - joblib
  - sqlalchemy
  - ssl
  - pickle
  - sklearn
  - text_length_extractor
  - sys
  - re

Please ensure that these libraries are installed on your system before running the code. If any of these libraries are missing, you can install them using pip. For example, to install the pandas library, you can use the following command:

```
pip install pandas
```

Note that some of these libraries have already been imported multiple times in the code, so please ensure that there are no duplicates.

- Navigate to the data folder and run python process_data.py. This will read in the disaster_messages.csv and disaster_categories.csv files, clean the data, and save the resulting data in a SQLite database called cleaned_data_sql.db.
  - Here is an example to run the program:
    - <img width="1125" alt="Bildschirm­foto 2023-02-26 um 10 30 53" src="https://user-images.githubusercontent.com/119667336/221402750-f46597e4-27a4-4392-9c05-551adc8513d4.png">
- Navigate to the models folder and run python train_classifier.py. This will read in the cleaned data from the cleaned_data_sql.db file, process it using a machine learning pipeline (using bow, pos, tf-idf, lemmatization), and train a random forest.
- Navigate to the app folder and run python run.py. This will start the web app.
