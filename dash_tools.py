import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import all_functions as af
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from bs4 import BeautifulSoup


df = pd.read_csv("top_250.csv")
new_X = af.remove_digits_tiny(df)
df_vectorized = af.vectorize_df(new_X)
term_frequency = df_vectorized.drop('budget', axis = 1).sum(axis=0).sort_values(ascending=False)

fig = go.Figure(data=[go.Bar(x = term_frequency[:100].index, y = term_frequency[:100].values)])

fig.update_layout(
            title = 'Distribution of Word frequency (Top 100)',
            yaxis={'title': 'Words'},
            xaxis = {'title': "Frequency"},
            margin=dict(l=50, r=15, b=40, t=10, pad=2),
        )

app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.Div([
        html.H1(
            children="Dashboard prediction box office",
            style={
                "textAlign": 'center',
                "margin": "30px"
            }
        ),
        html.Div([
            html.Div(
                    dcc.Graph(id='graph_pays', figure = fig), className="col-11 p-0 m-1 card shadow-sm rounded-0"),
        ]),
    ]),
])


@app.callback(Output('graph_pays', 'figure'))
def create_graph_pays():

    term_frequency = df_vectorized.drop('budget', axis = 1).sum(axis=0).sort_values(ascending=False)
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x = term_frequency[:100].index,
        y = term_frequency[:100].values,
        marker_color = "blue"
    ))

    fig.update_layout(
            title = 'Distribution of Word frequency (Top 100)',
            yaxis={'title': 'Words'},
            xaxis = {'title': "Frequency"},
            margin=dict(l=50, r=15, b=40, t=10, pad=2),
        )
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)