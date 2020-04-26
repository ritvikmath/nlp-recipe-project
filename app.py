import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from tensorflow.keras.models import load_model
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer

import pandas as pd

import pickle

# define variables
tabtitle='NLP Recipe Project'
myheading='Enter an Ingredient String'
default_ingredient = '2 tbsp butter'


# read model files

model = load_model('ingredients_model.h5')
y_dict = pickle.load(open("y_dict.p","rb"))
tokenizer_obj = pickle.load(open("tokenizer_obj.p","rb"))

max_length = 54 #this has been hard coded, likely it shouldnt be


# define app components

ingredient_input = dcc.Input(
    id="ingredient_input",
    type="text",
    placeholder="Enter Ingredient String", 
    style={'textAlign': 'center', 'width': '50%'},
    value=default_ingredient
)

submit_button = html.Button('Submit', id='submit_ingredient_string', n_clicks=0)

ingredient_output = html.Label(id='ingredient_output', children='butter', style={'fontSize': '24px'})

# initiate app

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title=tabtitle

# set layout
app.layout = html.Div([
        html.Div([
             html.H1(myheading),
             ingredient_input
             ], style={'padding':15}),
         html.Div([
             submit_button
             ]),
         html.Div([
             html.Label("Predicted Ingredient:", style={'fontSize': '20px'}),
             ingredient_output
             ], style={'padding':15, 'width': '70%', 'margin-left': 'auto', 'margin-right': 'auto'})
    ], style={'textAlign': 'center'}
)

@app.callback(
    dash.dependencies.Output('ingredient_output', 'children'),
    [dash.dependencies.Input('submit_ingredient_string', 'n_clicks')],
    [dash.dependencies.State('ingredient_input', 'value')])
def update_predicted_ingredient(n_clicks, value):
    
    try:
        test_sample = [value]
        test_sample_pad = pad_sequences(tokenizer_obj.texts_to_sequences(test_sample),maxlen=max_length,padding='post')
        test_sample_predict = model.predict(x=test_sample_pad)
        
        max_index = [list(p).index(max(p)) for p in test_sample_predict]
        predictions = y_dict[y_dict.index.isin(max_index)]["value"].reset_index()
        pred = pd.DataFrame({'text':list(test_sample),'predicted_id':max_index}).merge(predictions, left_on='predicted_id', right_on='index').value.iloc[0]
        
        return pred
        
    except:
        return None

if __name__ == '__main__':
    app.run_server()
