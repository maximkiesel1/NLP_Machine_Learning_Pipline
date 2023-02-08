# import libraries
import pandas as pd

# load data
df1 = pd.read_csv('/Users/maximkiesel/NLP_Pipline_disaster_response/data/disaster_messages.csv')
df2 = pd.read_csv('/Users/maximkiesel/NLP_Pipline_disaster_response/data/disaster_categories.csv')

df2.head()

# cleaning the row 'genre'
df1 = df1.replace(r"\\", "", regex=True)


# cleaning the column name
df1.rename(columns={'genre\\': 'genre'}, inplace=True)
df2.rename(columns={'categories\\': 'categories'}, inplace=True)

df2.isnull().sum()
