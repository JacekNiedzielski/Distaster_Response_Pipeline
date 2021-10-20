import json
import plotly
import pandas as pd


from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request
import plotly.graph_objects as g
import joblib
from sqlalchemy import create_engine

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///disaster_database.db')
df = pd.read_sql_table('Messages', engine)

# load model
model = joblib.load("class.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    #Genre counts:
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    fig_1 = {
            'data': [
                g.Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    
    
    #Occurence frequency of each category within the whole data set
    category_names = list(df.columns)
    category_names = category_names[3:]
    category_ratios = list()
    
    for category in category_names:
        ratio = (df[category].sum() / len(df))*100
        category_ratios.append(ratio)
     
    
    data = [g.Bar(y=category_names, x=category_ratios, orientation = "h")]

    layout = {  'height':600,
                'title': {'text':'Occurence frequency of each category within the whole data set',
                          'x':0.5, 'y':0.9,'xanchor': 'center', 'yanchor' : 'top'},
                'xaxis': {'tick0' : 5, 'dtick' : 5,
                    'title': "Percentual occurence [%]"},
                'yaxis': {'tickvals': category_names, 'tickmode': "array"}
            }
        
    fig_2 = g.Figure(data = data, layout = layout) 

    
    #Average length message length per category:
    average_message_lengths = list() 
    
    for category in category_names:
        total_length = 0
        for index, row in df[category].iteritems():
            if row == 1:
                total_length += len(df['message'][index])
            else:
                pass
        
        average_length = round(total_length / len(df.loc[df[category] == 1]))
        average_message_lengths.append(average_length)
        

    data = [g.Bar(y=category_names, x=average_message_lengths, orientation = "h")]

    layout = {  'height':600,
                'title': {'text':'Average message length per category',
                          'x':0.5, 'y':0.9,'xanchor': 'center', 'yanchor' : 'top'},
                'xaxis': {'tick0' : 0, 'dtick' : 10,
                    'title': "Average length [characters]"},
                'yaxis': {'tickvals': category_names, 'tickmode': "array"}
            }
        
    fig_3 = g.Figure(data = data, layout = layout)


    graphs = [fig_1, fig_2, fig_3]
    
        
    # Encoding plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Rendering web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


#Web page that handles user query and displays model results
@app.route('/go')
def go():
    #Saving user input in query
    query = request.args.get('query', '') 

    #Making predictions
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[3:], classification_labels))

    #This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()