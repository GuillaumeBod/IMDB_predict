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
df_ratings = pd.read_csv("ratings.csv")
df_predict_ratings = pd.read_csv("prediction.csv")



term_frequency = df_vectorized.drop('budget', axis = 1).sum(axis=0).sort_values(ascending=False)

fig_distribution = go.Figure(data=[go.Bar(x = term_frequency[1:100].index, y = term_frequency[1:100].values)])
fig_distribution.update_layout(
            title = 'Distribution of Word frequency (Top 100)',
            yaxis={'title': 'Frequency'},
            margin=dict(l=50, r=15, b=100, t=30, pad=2),
        )

fig_boxplot = go.Figure()
fig_boxplot.add_trace(go.Box(y=df_ratings["ratings"], name="Ratings not predicted"))
fig_boxplot.update_layout(
            margin=dict(l=30, r=30, b=100, t=30, pad=2),
        )

fig_boxplot_predict = go.Figure()
fig_boxplot_predict.add_trace(go.Box(y=df_predict_ratings["note_predites"], name="Ratings predicted"))
fig_boxplot_predict.update_layout(
            margin=dict(l=30, r=30, b=100, t=30, pad=2),
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
                    dcc.Graph(id='graph_distribution', figure = fig_distribution), className="col-11 p-0 m-1 card shadow-sm rounded-0"),
            
        ]),
        html.Div([
            html.Div(
                        dcc.Graph(id='graph_boxplot', figure = fig_boxplot), className="col-4 p-0 m-1 card shadow-sm rounded-0"),
            html.Div(
                        dcc.Graph(id='graph_boxplot_predicted', figure = fig_boxplot_predict), className="col-4 p-0 m-1 card shadow-sm rounded-0"),
        ],className="row justify-content-md-center", style={"display": "flex", "flexWrap": "wrap", "backgroud-color": "grey"}),
    ]),
])



if __name__ == '__main__':
    app.run_server(debug=True)