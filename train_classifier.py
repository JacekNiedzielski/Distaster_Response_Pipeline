import sys
#Natural Language Processing
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

#Data Science Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Regex and pickle
import re
import pickle

#Database sql connectivity
from sqlalchemy import create_engine

#Natural Language Processing Libraries
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

#Machine Learning Libraries (sklearn)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report



def load_and_split_data(database_filepath, table = "Messages", size = 0.1, random_state = 42):
    """
    This functions loads the data from the database and splits it to train
    and test set 

    Parameters
    ----------
    database_filepath:
        filepath of the database
    table:
        table to consider from the database
    size:
        size of the test data (in terms of ratio to the whole data ex: 0.2)
    random_state: 
        fixing the split to random state
    

    Returns
    -------
    X_train : Matrix
        Descriptive train data
    X_test : Matrix
        Descriptive test data
    y_train : Matrix
        Training labels
    y_test : Matrix
        Test labels
    X: Descriptive data (whole)
        
    Y: Labels (whole)
     
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table(table, engine)
    #General Division of Data
    X = df.iloc[:, 1]
    Y = df.iloc[:, 3:]
    #Train - test split
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = size)
    return (X_train, X_test, y_train, y_test, X, Y)


def tokenize(text):
    """
    Tokenizer functions which takes the text and performs series of 
    transformations:
        Normalization
        Toeknization using word_tokenize from nltk
        Lemmatization
    
    The function returns clean tokens
    """
    #Normalization - lowercase and punctuation removal:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    #Tokenization:
    words = text.split()
    words = word_tokenize(text)
    #Stop words removal:
    words = [w for w in words if w not in stopwords.words("english")]
    #Lemmatization
    lemmed = [WordNetLemmatizer().lemmatize(w, pos = "v") for w in words]
    
    return lemmed

def make_sklearn_pipeline(Pipeline = Pipeline, memory = None, verbose = False, **kwargs):
    """
    Firt argument is sklearn Pipepline class. It should not be changed.
    
    Definition of pipeline steps happens EXPLICITLY within the instantiation!
    Definition of typical pipeline looks like following:
    example_pipeline = make_sklearn_pipeline(steps = [('name#1',transformer#1),
                                                      ('name#2',transformer#2),
                                                      ('name#3',transformer#3),
                                                      ('name#4',transformer#4),
                                                      ........................,
                                                      ('name#n',classifier#n)], 
                                                      verbose = ...,
                                                      memory = ...)
    
    It is not required to give values for ´verbose´ and ´memory´. 
    They have default values as False and None respectively. 
    For more information visit: 
    ´https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html´
    """
    
    pipeline = Pipeline(steps = kwargs["steps"], memory=memory, verbose = verbose)
    
    return pipeline
    
    
    
def df_from_sklearn_cl_reports(cl_report):
    """
    This function transforms the sklearn_cl_report into a simple data frame
    Thanks to this procedure the results for different categories are easier 
    to comparte

    Parameters
    ----------
    cl_reports : sklearn classification report object

    Returns: Data Frame with precisions, recalls and f1_scores from the sklearn
    classification report
    """
    data_frame = pd.DataFrame()
    #This for loop extracts labels, precisions, recalls and f1_scores.
    #List expressions inside the loop are just the trival transformations
    #of the classification report
    for feature in list(cl_report.keys()):
        #Preparing lists for future data series
        labels = []
        precisions = []
        recalls = []
        f1_scores = []
        #List comprehensions are used to form one main list of results
        l = cl_report[feature].split(' ')
        l = [x for x in l if x !='']
        l = [x for x in l if '\n' not in x]
        l = l[:l.index("accuracy")]
        columns = l[:3]
        l = [x for x in l if x not in columns]
        #Each fourth element in the list "l" refers to the labels, precisions, 
        #recalls and f1_scores respectivelly
        for i, element in enumerate(l):
            if i == 0 or i % 4 == 0:
                labels.append(element)
                precisions.append(l[i+1])
                recalls.append(l[i+2])
                f1_scores.append(l[i+3])
        #Communicates are simply our labels. We obtain them by pasting the
        #features name at each index
        communicates = [feature]*len(labels)
        #Formation of the data frame:
        #Beginning:
        if data_frame.shape[0] == 0:
            data_frame["communicate"] = communicates
            data_frame["label"] = labels
            data_frame["precisions"] = precisions
            data_frame["recalls"] = recalls
            data_frame["f1_scores"] = f1_scores
        #If already some categories are given:
        else:
            auxilliary_df = pd.DataFrame()
            auxilliary_df['communicate'] = communicates
            auxilliary_df['label'] = labels
            auxilliary_df['precisions'] = precisions
            auxilliary_df['recalls'] = recalls
            auxilliary_df['f1_scores'] = f1_scores
            
            data_frame = pd.concat([data_frame, auxilliary_df])
            
            del auxilliary_df
            
    data_frame.set_index(["communicate"], inplace = True)
    
    #Changing the data types
    data_frame.label = data_frame.label.astype("float")
    data_frame.label = data_frame.label.astype("int")
    data_frame.precisions = data_frame.precisions.astype("float")
    data_frame.recalls = data_frame.recalls.astype("float")
    data_frame.f1_scores = data_frame.f1_scores.astype("float")
            
    return data_frame    
        
        


def build_model():
    """
    This function makes use of the 'make_sklearn_pipeline' function and
    returns the trained model
    """
    model = make_sklearn_pipeline(
    verbose = True, steps = [('vect', CountVectorizer(tokenizer = tokenize)),
                             ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(estimator = DecisionTreeClassifier
               (max_depth=7, min_samples_leaf=1, class_weight = "balanced", random_state = 42)))])

    return model




def evaluate_model(model, X_test, y_test, Y):
    """
    Prints classification report, mean precision over all categories and
    mean recall over all categories
    
    Parameters
    ----------
    model : Instance of machine learning model
    X_test : Test data (features)
    y_test : Test data (labels)
        
    Y : All labels - just for purposes of itteration over all columns

    This function does not return anything
    -------
    """
    classification_reports = {}
    
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns = list(Y.columns))
    y_test = y_test.reset_index(drop = True)
    for i, var in enumerate(Y):
        print(var)
        print(classification_report(y_test.iloc[:,i], y_pred.iloc[:, i]))
        
    for i,var in enumerate(Y):
        classification_reports[var] = (classification_report(
            y_test.iloc[:,i], y_pred.iloc[:,i]))
        
    report = df_from_sklearn_cl_reports(classification_reports)
    
    mean_precision = report.loc[report.label == 1].groupby(
        ["communicate"])["precisions"].mean().mean()
    mean_recall = report.loc[report.label == 1].groupby(
        ["communicate"])["recalls"].mean().mean() 
    
    classification_reports.clear()    
    
    print("Mean precision for all categories: ", mean_precision)
    print("Mean recall for all categories: ", mean_recall)    
        
def save_model(model, model_filepath):
    """
    This function saves the trained model as a pickle file. Specified is the 
    model itself and wished filepath under which it should be stored

    Parameters
    ----------
    model : The trained ML model 
        
    model_filepath : Filepath to store the pickle file
    
    """
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    """
    Main function of the program - takes no parameters
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X_train, X_test, y_train, y_test, X, Y = load_and_split_data(database_filepath)
        #print(X_train)
        #print("\n")
        #print(X_test)
        #print("\n")
        #print(y_train)
        #print("\n")
        #print(y_test)
        #print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, Y)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()