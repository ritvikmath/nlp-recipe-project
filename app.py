import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

import pickle
from random import sample


#create initial strings              
tabtitle='NLP Recipe Project'
landing_header_string = 'Select Recipes You Like!'
recipes_selected_string = '0 Recipes Selected'
done_string = 'Done'
refresh_string = 'Refresh'
reset_string = 'Reset'
link_string = 'Link'

rec_header_string = 'Your Recommendations'
return_string = 'Return'
bow_string= 'Bag of Words'
doc2vec_string = 'Doc2Vec'
lstm_string = 'LSTM'
bert_string = 'BERT'

#this global will keep track of recs
app_recs = {'BoW': [], 'D2V': [], 'LSTM': [], 'BERT': []}

#these lists define the html element ids
available_button_ids = ['tl', 'tc', 'tr', 'bl', 'bc', 'br']
available_card_ids = [bid + '_card' for bid in available_button_ids]
available_title_ids = [bid + '_title' for bid in available_button_ids]
available_url_ids = [bid + '_url' for bid in available_button_ids]

#the default style for a recipe card
base_style = {"width": "90%", 'align': 'center', 'background-color': '#f0f0f0'}
selected_style = {"width": "90%", 'align': 'center', 'background-color': '#d2f2cb'}
              
#this global will keep track of which recipes are currently "liked"              
current_liked_recipe_ids = []

#this global keeps track of all the info regarding each card
current_input_state = {bid: {'id':'', 'title':'', 'url':'', 'style': base_style, 'picked':False} for bid in available_button_ids}

#read data file
recs_dict = pickle.load(open("recs_file.p","rb"))
id_to_info = recs_dict['id_to_info']
id_to_recs = recs_dict['id_to_recs']
all_recipe_ids = list(id_to_info.keys())

#this function generates 6 random recipes and updates the global current_input_state accordingly
def refresh_cards():
    global current_input_state
    
    chosen_recipe_ids = sample(all_recipe_ids, 6)
    current_recipe_info = [(rid,) + id_to_info[rid] for rid in chosen_recipe_ids]
    
    for idx, r_info in enumerate(current_recipe_info):
        bid = available_button_ids[idx]
        current_input_state[bid]['id'] = r_info[0]
        current_input_state[bid]['title'] = r_info[1]
        current_input_state[bid]['url'] = r_info[2]
        current_input_state[bid]['picked'] = False
        current_input_state[bid]['style'] = base_style

#start off by getting 6 random recipes
refresh_cards()

# initiate app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title=tabtitle 
 
#this function creates a recipe card with a given title, url, and button id
def create_recipe_card(recipe_title, recipe_url, button_id):

    return dbc.Card(
        dbc.CardBody(
            [
                html.H1(recipe_title, id=button_id + '_title', className="card-title", style={'font-size':'20px', 'textAlign': 'center'}),
                html.Div([dbc.CardLink("Recipe Link", id=button_id + '_url', href=recipe_url, external_link=True, target="_blank", style={'font-size':'18px'})], style={'textAlign': 'center', 'padding': '1%'}),
                html.Div([dbc.Button("I Like This!", id=button_id, color="success", style={'font-size':'20px'})], style={'textAlign': 'center', 'padding-top': '3%'})
            ]
        ),
        id = button_id + '_card',
        style={"width": "90%", 'align': 'center', 'background-color': '#f0f0f0'}
    )

#set the layout             
app.layout = dbc.Container(
    [
        html.H1(landing_header_string, style={'textAlign': 'center', 'padding-top': '3%', 'padding-bottom': '3%', 'font-size': '40px'}),
        
        dbc.CardDeck(
            [
                create_recipe_card(current_input_state['tl']['title'], current_input_state['tl']['url'], 'tl'),
                create_recipe_card(current_input_state['tc']['title'], current_input_state['tc']['url'], 'tc'),
                create_recipe_card(current_input_state['tr']['title'], current_input_state['tr']['url'], 'tr')
            ],
            style={'padding-bottom': '1%'}
        ),
        dbc.CardDeck(
            [
                create_recipe_card(current_input_state['bl']['title'], current_input_state['bl']['url'], 'bl'),
                create_recipe_card(current_input_state['bc']['title'], current_input_state['bc']['url'], 'bc'),
                create_recipe_card(current_input_state['br']['title'], current_input_state['br']['url'], 'br')
            ],
            style={'padding-bottom': '1%'}
        ),
        dbc.Row([
            dbc.Col(html.H1(recipes_selected_string, id='num_recipes', style={'textAlign': 'center', 'font-size': '25px'}), width={'size': 4, 'offset': 4})
            ]),
        dbc.Row([
                dbc.Col(html.Div([dbc.Button("Refresh", id="refresh", color="primary", style={'font-size': '20px'})], style={'textAlign': 'center', 'padding-top': '5%'})),
                dbc.Col(html.Div([dbc.Button("Reset", id="reset", color="danger", style={'font-size': '20px'})], style={'textAlign': 'center', 'padding-top': '5%'})),
                dbc.Col(html.Div([dbc.Button("Done", id="done", color="success", style={'font-size': '20px'})], style={'textAlign': 'center', 'padding-top': '5%'}))
            ])
    ],
    fluid=True
)
    
#change global vars if reset    
def reset_cards():
    #use globals
    global current_input_state
    global current_liked_recipe_ids
    
    current_liked_recipe_ids = []
    
    for bid in current_input_state:
        current_input_state[bid]['picked'] = False
        current_input_state[bid]['style'] = base_style
        
#this function generates recs based on current liked recipes
def get_recs():
    pass

#this function changes a chard if it has been clicked on
def change_card(trigered_element_id):
    global current_input_state
    global current_liked_recipe_ids
    
    if not current_input_state[trigered_element_id]['picked']:
        current_input_state[trigered_element_id]['style'] = selected_style
        current_liked_recipe_ids.append(current_input_state[trigered_element_id]['id'])
        current_input_state[trigered_element_id]['picked'] = True

#this function gets the current state of all global info vars and makes a formatted callback output list        
def generate_callback_output():
    global current_input_state
    global current_liked_recipe_ids
    
    card_styles = [current_input_state[i]['style'] for i in available_button_ids]
    titles = [current_input_state[i]['title'] for i in available_button_ids]
    urls = [current_input_state[i]['url'] for i in available_button_ids]
    
    recipes_selected_string = ['%s Recipes Selected'%len(current_liked_recipe_ids)]
    
    output = card_styles + titles + urls + recipes_selected_string
    
    return output

#this callback accepts as input the following elements:
#[all 'I Like This!" buttons ... rest button ... refresh button ... done button] 
        
#this callback returns as output the following elements:
#[all card styles ... all card titles ... all card urls ... current number of recipes selected string] 
@app.callback(
    [dash.dependencies.Output(cid, 'style') for cid in available_card_ids] + 
    [dash.dependencies.Output(tid, 'children') for tid in available_title_ids] +
    [dash.dependencies.Output(uid, 'href') for uid in available_url_ids] +
    [dash.dependencies.Output('num_recipes', 'children')],
     
    [dash.dependencies.Input(bid, 'n_clicks') for bid in available_button_ids] +
    [dash.dependencies.Input('reset', 'n_clicks')] + 
    [dash.dependencies.Input('refresh', 'n_clicks')] + 
    [dash.dependencies.Input('done', 'n_clicks')])
def update_app_upon_liked_recipe(*n_clicks_vals):
    
    #get the current callback context    
    ctx = dash.callback_context

    #if nothing changed, just return
    if not ctx.triggered:
        return

    #get the element id who triggered this update
    triggered_element_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    #if this was the refresh
    if triggered_element_id == 'refresh':
        
        refresh_cards()
            
    #if this was the reset
    elif triggered_element_id == 'reset':
        
        reset_cards()
     
    #if this was the done
    elif triggered_element_id == 'done':
        
        get_recs()
        
    #otherwise this was a "I Like This!" button
    else:
        
        change_card(triggered_element_id)
        
    output = generate_callback_output()
    
    return output

if __name__ == '__main__':
    app.run_server()

