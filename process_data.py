import sys
#Data Science Libraries
import pandas as pd
import numpy as np
#Regex
import re
#Database sql connectivity
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = pd.merge(messages, categories, on = "id")
    
    return df
    

def clean_data(df, separator = ";"):
    """
    Parameters
    ----------
    df : Main DataFrame
    separator : Categories separator (default = ';')
    Returns: cleaned data frame -> df
    -------
    """
    
    #Categories:
    categories = df.categories.str.split(";", expand = True)
    
    #Renaming categories columns:
    row = categories.iloc[[0]]
    
    for i, element in row.copy().iteritems():
        
        new_element = re.sub(r"[^a-zA-Z_]","",element[0])
        row[i] = new_element
        
    category_colnames = row.values[0]
    categories.columns = category_colnames
    
    #Converting category values to zeros and ones

    for column in categories:
        categories[column] = categories[column].str[-1:]
        categories[column] = categories[column].astype("int")
        
      
    
    #Double check if there are only zeros and ones:
        
    def check_for_true_false(df, threshold = 0.05):
        """
        This function checks if there are only ones and zeros given in 
        the data frame
    
        Parameters
        ----------
        df : considered data frame.
        threshold : Sets the maximum ratio between all rows and rows with True
        and False only which allows to simply drop the rows with other values
        The default is 0.05. (5%)

        Returns considered data frame -> df
        -------
        """
        #Just to be sure to iterrate always over the same data frame it is 
        #justified to make an extra copy for the "for loop":
            
        df2 = df.copy()
        
        for category in df:
            
            if len(df2[category].value_counts().index) == 1:
                print("category {} not useful for predictions\
since only one possible value is given".format(category))
                df.drop(columns = category, axis = 1, inplace = True)
                print("\n")
                
            if len(df2[category].value_counts().index) > 2:
                #Data Frame with zeros and ones only:
                tf_only = df[category].value_counts().sort_index()[:2]
                #Data Frame with all values:
                all_values = df[category].value_counts().sort_index()
                #Checking the ration of tf_values to all values
                #Is the ratio below threshold, then we can drop the excess
                tf_only_sum = tf_only.sum()
                all_values_sum = all_values.sum()
                ratio = 1 - (tf_only_sum / all_values_sum)
                print("Too many labels in category {}!".format(category))
                
                print("The number of all rows is {}\
, the number of rows with True or False only is {} -> hence the \
ratio of excessive classes values makes {}".format(all_values_sum, 
                                                  tf_only_sum,
                                                  ratio))
                if ratio < threshold:
                    for value in list(all_values.index)[2:]:
                        df.replace(value, np.nan, inplace = True)
                    df.dropna(subset = [category], inplace = True)
                    
                    print("Excessive labels have been dropped")      
                    print("\n")
        return df
    
    
    #Running the double check function on the categories
    check_for_true_false(categories)                
    
    
    #Dropping redundant columns from the data frame
    df.drop(columns = ["categories", "original"], 
                      axis = 1, inplace = True)
    
    #Concatinating all 36 categories to the
    df = pd.concat([df.copy(), categories], axis = 1)
    
    #Dropping duplicates
    df.drop_duplicates(inplace = True)
    
    #Dropping nan values
    df.dropna(inplace = True)
    print(df)
    
    #Checking if there are no singular classes (with only one value beeing
    #either True or False for all of the records)
    
    for column in df.iloc[:,3:]:
        if len(df[column].value_counts()) < 2:
            print(column+" has only one label")
            print("The feature will be removed!")
            df.drop(columns = column, axis = 1, inplace = True)
            
        else:
            print('{:10}'.format(column))
            print('{:>10}'.format("Contains both True and False values - Ok!"))
            print("\n")
    
    return df


def save_data(df, database_filename):
    
    engine = create_engine("sqlite:///"+database_filename)
    c = engine.connect()
    conn = c.connection
    df.to_sql('Messages', con = conn, index=False, if_exists = 'replace')
    
             

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
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
