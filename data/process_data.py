# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# load data
df1 = pd.read_csv('/Users/maximkiesel/NLP_Pipline_disaster_response/data/disaster_messages.csv')
df2 = pd.read_csv('/Users/maximkiesel/NLP_Pipline_disaster_response/data/disaster_categories.csv')

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

# get the column name of the target variable
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
concat_df = pd.concat([df1, y], axis=1)

# delete dulicates
concat_df.drop_duplicates(inplace=True)

# create a sql database
engine = create_engine('sqlite:///cleaned_data_sql.db')
concat_df.to_sql('clean_disaster_messages', engine, index=False)