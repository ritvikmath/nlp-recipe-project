import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from urllib.request import urlopen

import pickle
from random import sample
import requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


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

#current mode is landing
curr_mode = 'landing'

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

button_rec_style = {'font-size':'18px', 'display': 'none'}
button_input_style = {'font-size':'18px'}

inputs_button_style = {'font-size': '20px'}
inputs_button_style_hidden = {'font-size': '20px', 'display':'none'}

num_recipes_style = {'textAlign': 'center', 'font-size': '25px'}
num_recipes_style_hidden = {'textAlign': 'center', 'font-size': '25px', 'display':'none'}

dropdown_style = {'font-size': '30px'}
dropdown_style_hidden = {'font-size': '30px', 'display':'none'}
              
#this global will keep track of which recipes are currently "liked"              
current_liked_recipe_ids = []

#this global keeps track of all the info regarding each card
current_input_state = {bid: {'id':'', 'title':'', 'url':'', 'style': base_style, 'button_style': button_input_style, 'picked':False} for bid in available_button_ids}

#this global keeps track of the current model name
curr_model_label = 'BoW'


#read data file
try:
    recs_dict = pickle.load(open("populated_recs_dict.p","rb"))
except FileNotFoundError:
    file_id = '11FabBVXrMURck-i06S1vSXXvkMQ6UGDM'
    download_file_from_google_drive(file_id, 'gdrive_populated_recs_dict.p')
    recs_dict = pickle.load(open("gdrive_populated_recs_dict.p","rb"))
 
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
        current_input_state[bid]['button_style'] = button_input_style

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
                html.Div([dbc.Button("I Like This!", id=button_id, color="success", style=button_input_style)], style={'textAlign': 'center', 'padding-top': '3%'})
            ]
        ),
        id = button_id + '_card',
        style={"width": "90%", 'align': 'center', 'background-color': '#f0f0f0'}
    )

#create the model selection dropdown
model_dropdown = dbc.DropdownMenu(
    label="Model",
    id='model',
    bs_size="lg",
    className="mb-3",
    children=[
        dbc.DropdownMenuItem("Bag of Words", id='BoW', style={'font-size': '16px'}),
        dbc.DropdownMenuItem("Doc2Vec", id='D2V', style={'font-size': '16px'}),
        dbc.DropdownMenuItem("LSTM", id='LSTM', style={'font-size': '16px'}),
        dbc.DropdownMenuItem("BERT", id='BERT', style={'font-size': '16px'}),
    ],
    direction='down',
    style=dropdown_style_hidden
)

#set the layout     
        
app.layout = dbc.Container(
    [
        html.H1(landing_header_string, id='header_string', style={'textAlign': 'center', 'padding-top': '3%', 'padding-bottom': '3%', 'font-size': '40px'}),
        
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
                dbc.Col(html.Div([dbc.Button("Back", id="back", color="secondary", style=inputs_button_style_hidden)], style={'textAlign': 'center', 'padding-top': '5%'})),
                dbc.Col(html.H1(recipes_selected_string, id='num_recipes', style=num_recipes_style)),
                dbc.Col(html.Div([model_dropdown], style={'textAlign': 'center', 'padding-top': '0'}))
            ]),
        dbc.Row([
                dbc.Col(html.Div([dbc.Button("Refresh", id="refresh", color="primary", style=inputs_button_style)], style={'textAlign': 'center', 'padding-top': '5%'})),
                dbc.Col(html.Div([dbc.Button("Reset", id="reset", color="danger", style=inputs_button_style)], style={'textAlign': 'center', 'padding-top': '5%'})),
                dbc.Col(html.Div([dbc.Button("Done", id="done", color="success", style=inputs_button_style)], style={'textAlign': 'center', 'padding-top': '5%'}))
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
    global current_liked_recipe_ids
    global app_recs
    global current_input_state
    global curr_mode
    
    #get all recs
    for rid in current_liked_recipe_ids:
        app_recs['BoW'].extend(id_to_recs[rid]['BoW'])
        app_recs['D2V'].extend(id_to_recs[rid]['D2V'])
        app_recs['LSTM'].extend(id_to_recs[rid]['LSTM'])
        app_recs['BERT'].extend(id_to_recs[rid]['BERT'])
    
    #just sample six from each model
    for model in ['BoW', 'D2V', 'LSTM', 'BERT']:
        app_recs[model] = sample(app_recs[model], 6)
        
    for idx,bid in enumerate(current_input_state):
        #all cards not chosen
        current_input_state[bid]['picked'] = False
        #all cards have base style
        current_input_state[bid]['style'] = base_style
        #all "I Like This!" buttons are hidden
        current_input_state[bid]['button_style'] = button_rec_style
        #the initial set of recs will be from BoW
        info = id_to_info[app_recs['BoW'][idx]]
        current_input_state[bid]['title'] = info[0]
        current_input_state[bid]['url'] = info[1]
     
    #switch the current mode
    curr_mode = 'recs'
    
def go_back():
    global current_liked_recipe_ids
    global app_recs
    global current_input_state
    global curr_mode
    global curr_model_label
    
    curr_model_label = 'BoW'
    
    refresh_cards()
    app_recs = {'BoW': [], 'D2V': [], 'LSTM': [], 'BERT': []}
    current_liked_recipe_ids = []
    curr_mode = 'landing'
    
    
  
def change_model(model_name):
    global app_recs
    global current_input_state
    global curr_model_label
    
    curr_model_label = model_name
    
    for idx,bid in enumerate(current_input_state):
        info = id_to_info[app_recs[model_name][idx]]
        current_input_state[bid]['title'] = info[0]
        current_input_state[bid]['url'] = info[1]
    
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
    global curr_mode
    global curr_model_label
    
    button_styles = [current_input_state[i]['button_style'] for i in available_button_ids]
    card_styles = [current_input_state[i]['style'] for i in available_button_ids]
    titles = [current_input_state[i]['title'] for i in available_button_ids]
    urls = [current_input_state[i]['url'] for i in available_button_ids]
    
    recipes_selected_string = ['%s Recipes Selected'%len(current_liked_recipe_ids)]
    
    recipes_selected_style = [num_recipes_style] if curr_mode == 'landing' else [num_recipes_style_hidden]
    
    header_string = [landing_header_string if curr_mode == 'landing' else rec_header_string]
    
    input_buttons_styles = [inputs_button_style]*3 if curr_mode == 'landing' else [inputs_button_style_hidden]*3
    
    back_button_style = [inputs_button_style_hidden] if curr_mode == 'landing' else [inputs_button_style]
    
    model_dropdown_style = [dropdown_style_hidden] if curr_mode == 'landing' else [dropdown_style]
    
    model_label = [curr_model_label]
    
    output = button_styles + card_styles + titles + urls + recipes_selected_string + recipes_selected_style + header_string + input_buttons_styles + back_button_style + model_dropdown_style + model_label
    
    return output

#this callback accepts as input the following elements:
#[all 'I Like This!" buttons ... rest button ... refresh button ... done button] 
        
#this callback returns as output the following elements:
#[all button styles ... all card styles ... all card titles ... all card urls ... current number of recipes selected string] 
@app.callback(
    [dash.dependencies.Output(bid, 'style') for bid in available_button_ids] + 
    [dash.dependencies.Output(cid, 'style') for cid in available_card_ids] + 
    [dash.dependencies.Output(tid, 'children') for tid in available_title_ids] +
    [dash.dependencies.Output(uid, 'href') for uid in available_url_ids] +
    [dash.dependencies.Output('num_recipes', 'children')] +
    [dash.dependencies.Output('num_recipes', 'style')] +
    [dash.dependencies.Output('header_string', 'children')] + 
    [dash.dependencies.Output(inputs_button_id, 'style') for inputs_button_id in ['reset', 'refresh', 'done']] +
    [dash.dependencies.Output('back', 'style')] +
    [dash.dependencies.Output('model', 'style')] + 
    [dash.dependencies.Output('model', 'label')],
     
    [dash.dependencies.Input(bid, 'n_clicks') for bid in available_button_ids] +
    [dash.dependencies.Input('reset', 'n_clicks')] + 
    [dash.dependencies.Input('refresh', 'n_clicks')] + 
    [dash.dependencies.Input('done', 'n_clicks')] + 
    [dash.dependencies.Input('back', 'n_clicks')] +
    [dash.dependencies.Input(model_name, 'n_clicks') for model_name in ['BoW', 'D2V', 'LSTM', 'BERT']])
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
        
    #if this was a dropdown model selection
    elif triggered_element_id in ['BoW', 'D2V', 'LSTM', 'BERT']:
        
        change_model(triggered_element_id)
    
    #if this was the back button
    elif triggered_element_id == 'back':
        
        go_back()
        
    #otherwise this was a "I Like This!" button
    else:
        
        change_card(triggered_element_id)
        
    output = generate_callback_output()
    
    return output

if __name__ == '__main__':
    app.run_server()

