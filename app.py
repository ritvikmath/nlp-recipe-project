import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import requests
import dash_table
import re

########### Define your variables
tabtitle='NLP Recipe Project'
myheading='Enter a Recipe URL'
recipe_link = 'https://www.allrecipes.com/recipe/20144/banana-banana-bread/'

common_measurements = ['cup', 'tablespoon', 'teaspoon']

def get_amount_measurement_ingredient(s):
    #get amount
    amount = re.findall(r'^(\d+(?:(?: \d+)*\/\d+)?)', s)
    amount = amount[0] if len(amount) > 0 else ''
    
    #get measurement
    measurement = ''
    for c in common_measurements:
        matches = re.findall(c + 's?', s)
        if len(matches) > 0:
            measurement += matches[0]
    
    #get the ingredient
    ingredient = s.replace(amount, '').replace(measurement, '').strip()
    
    return amount, measurement, ingredient

########### App components

recipe_input = dcc.Input(
    id="recipe_input",
    type="text",
    placeholder="Enter Recipe URL", 
    style={'textAlign': 'center', 'width': '50%'},
    value=recipe_link
)

submit_button = html.Button('Submit', id='submit_url', n_clicks=0)

recipe_output = dash_table.DataTable(
        id='recipe_output',
        columns=[{"name": "Amount", "id": 'amount'}, {"name": "Measurement", "id": 'measurement'}, {"name": "Ingredient", "id": 'ingredient'}],
        style_cell_conditional=[
        {'if': {'column_id': 'amount'},
         'textAlign': 'left'},
        {'if': {'column_id': 'measurement'},
         'textAlign': 'left'},
        {'if': {'column_id': 'ingredient'},
         'textAlign': 'left'},
    ]
    )

########### Initiate the app

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title=tabtitle

########### Set up the layout
app.layout = html.Div([
        html.Div([
             html.H1(myheading),
             recipe_input
             ], style={'padding':15}),
         html.Div([
             submit_button
             ]),
         html.Div([
             recipe_output
             ], style={'padding':15, 'width': '70%', 'margin-left': 'auto', 'margin-right': 'auto'})
    ], style={'textAlign': 'center'}
)

@app.callback(
    dash.dependencies.Output('recipe_output', 'data'),
    [dash.dependencies.Input('submit_url', 'n_clicks')],
    [dash.dependencies.State('recipe_input', 'value')])
def update_recipe_result(n_clicks, value):
    
    try:
        result = requests.get(value.strip()).text
        ingredients = re.findall(r'itemprop="recipeIngredient">(.*)<', result)
        parsed_ingredients = [get_amount_measurement_ingredient(s) for s in ingredients]
        table_data = [{'amount':p[0], 'measurement':p[1], 'ingredient':p[2]} for p in parsed_ingredients]
    except:
        return []
    
    return table_data

    

if __name__ == '__main__':
    app.run_server()
