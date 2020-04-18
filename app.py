import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

########### Define your variables
tabtitle='NLP Recipe Project'
myheading='Enter a Recipe URL'
recipe_link = 'https://www.allrecipes.com/recipe/20144/banana-banana-bread/'

########### App components

recipe_input = dcc.Input(
    id="recipe_input",
    type="text",
    placeholder="Enter Recipe URL"
)

recipe_output = html.Div(id="recipe_ouput")

########### Initiate the app

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title=tabtitle

########### Set up the layout
app.layout = html.Div(
        [recipe_input, recipe_output]
)

@app.callback(
    Output("recipe_output", "children"),
    Input("recipe_input", "value"),
)
def cb_render(val):
    return val

if __name__ == '__main__':
    app.run_server()
