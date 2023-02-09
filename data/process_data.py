# import libraries
import pandas as pd

# load data
df1 = pd.read_csv('/Users/maximkiesel/NLP_Pipline_disaster_response/data/disaster_messages.csv')
df2 = pd.read_csv('/Users/maximkiesel/NLP_Pipline_disaster_response/data/disaster_categories.csv')
pd.options.display.max_rows = None
df2.head()

# cleaning the row 'genre'
df1 = df1.replace(r"\\", "", regex=True)


# cleaning the column name
df1.rename(columns={'genre\\': 'genre'}, inplace=True)
df2.rename(columns={'categories\\': 'categories'}, inplace=True)

# get the column name of the target variable
list_col_target = [] # list for the target columns
for i in df2.iloc[0,1].split(';'): # loop through the category values
    col_name = i.split('-')[0]
    list_col_target.append(col_name)

    
# create a dummy dataframe for the target variable
y = pd.DataFrame(columns=list_col_target)

# dummy dictionary for the rows for y
dict = {name: 0 for name in list_col_target}
# fill the dataframe
for i in df2['categories'].values:
    y = y.append(dict, ignore_index=True)
    split1 = i.split(';')
    for j in split1:
        split2 = j.split('-')
        #print(split2)
        if '1' in split2:
            print(split2)
            y = y.append({split2[0]: 1}, ignore_index=True)
            print(y.head(1))
            break
            
    break