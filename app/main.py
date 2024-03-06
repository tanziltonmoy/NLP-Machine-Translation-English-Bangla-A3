import dash
from dash import dcc, html, Input, Output
from model import translate_sentence
import json
import torch  # Add this import

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server  # Required for deployment

# Load vocab and model configuration
vocab_transform = torch.load('./Models/vocab.pt')
with open('./Models/model_config.json', 'r') as config_file:
    loaded_config = json.load(config_file)

device = torch.device('cpu')  # Define device

params, state = torch.load('./Models/Additive_Attn_Seq2SeqTransformer.pt', map_location=device)
model = Seq2SeqTransformer(**params, device=device)
model.load_state_dict(state)
model.eval()  # Set the model to evaluation mode

# Define styles
external_stylesheets = ['https://cdnjs.cloudflare.com/ajax/libs/normalize/8.0.1/normalize.min.css',
                        'https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css',
                        {'href': 'assets/style.css', 'rel': 'stylesheet'}]

# Define app layout
app.layout = html.Div([
    html.H1("English-Bangla Translation", className="title"),
    
    # Help Documentation Section
    html.Div([
        html.H3("How to Use"),
        html.P("Enter an English sentence in the input box below and click the 'Translate Sentence' button."),
        html.P("The translated sentence will appear below the button.")
    ], className="help-doc"),
    
    html.Div(id='translation-output', className="translation-output"),
    dcc.Input(id='src_sentence', type='text', placeholder="Enter Sentence (English)...", className="input-box"),
    
    # Change Button Name
    html.Button('Translate Sentence', id='translate-button', n_clicks=0, className="btn-translate")
], className="container")

# Define callback to handle translation
@app.callback(
    Output('translation-output', 'children'),
    [Input('translate-button', 'n_clicks')],
    [Input('src_sentence', 'value')]
)
def translate_sentence_callback(n_clicks, sentence):
    if n_clicks > 0 and sentence:
        translation = translate_sentence(sentence, vocab_transform['en'], vocab_transform['bn'], model, device)
        return html.Div([
            html.H3("Bangla Translation:", className="translation-heading"),
            html.P(translation, className="translation-text")
        ])
    else:
        return html.Div()

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, external_stylesheets=external_stylesheets)
