# import libraries
import pandas as pd
from sqlalchemy import create_engine
import sys


# load data
def load_data(messages_filepath, categories_filepath):
    '''
    Load the csv data
        
    INPUT 
    messages_filepath - csv data with short text messages
    categories_filepath - csv data with the categorization of the data
        
    OUTPUT
    df1 - dataframe of the short text messages
    df2 - dataframe of the categorization of the data
    '''
    df1 = pd.read_csv(messages_filepath)
    df2 = pd.read_csv(categories_filepath)
    return df1, df2
    
# clean the data
def clean_data(df1, df2):
    '''
    Cleaning both dataframes and unit them to one
        
    INPUT 
    df1 - dataframe of the short text messages
    df2 - dataframe of the categorization of the data
        
    OUTPUT
    df - united dataframe 
    '''
    # cleaning the row 'genre'
    df1 = df1.replace(r"\\", "", regex=True)

    # cleaning the column name
    df1.rename(columns={'genre\\': 'genre'}, inplace=True)
    df2.rename(columns={'categories\\': 'categories'}, inplace=True)  
    
    # delete the last empty row. this empy row breaks the tranforming code of the target  columns
    df2 = df2.drop(26248)
    df1 = df1.drop(26248)

    # delete not necessary columns
    df1.drop(columns=['original', 'genre', 'id'], inplace=True)

    # get the column name of the target variables y
    list_col_target = [] # list for the target columns
    for i in df2.iloc[0,1].split(';'): # loop through the category values
        col_name = i.split('-')[0]
        list_col_target.append(col_name)
    
    # create a dummy dataframe for the target variable
    y = pd.DataFrame(columns=list_col_target)

    # dummy dictionary for the rows for y
    dummy_df = pd.DataFrame([{name: 0 for name in list_col_target}])

    # row counter
    row_count = -1

    # fill the y dataframe
    for i in df2['categories'].values:
        row_count += 1
        y = pd.concat([y,dummy_df], ignore_index=True)
        split1 = i.split(';')
        for j in split1:
            split2 = j.split('-')
            if '1' in split2:
                y.at[row_count, split2[0]] = split2[1]
            

    # connect both dfs togther 
    df = pd.concat([df1, y], axis=1)

    # delete duplicates
    df.drop_duplicates(inplace=True)
    
    return df

# create a sql database
def save_data(df, database_filepath):
    '''
    Save the dataframe in a sql database
        
    INPUT 
    df - united dataframe
    database_filepath - saving path
        
    OUTPUT
    None 
    '''
    engine = create_engine(f'sqlite:///{database_filepath}')
    df.to_sql('cleaned_data', engine, index=False)
    return None

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'.format(messages_filepath, categories_filepath))
        df1, df2 = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df1, df2)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()