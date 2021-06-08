import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd


class Server:
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    def __init__(self):
        self.app = app = dash.Dash(
            __name__, external_stylesheets=Server.external_stylesheets)

    def __call__(self, fig, name="Graph"):
        self.deploy_graph(fig, name)

    def deploy_graph(self, fig, name="Graph"):
        self.app.layout = html.Div(children=[
            html.H1(children='Dash Server'),
            html.Div(children='''
            Dash: A web application framework for Python.
        '''),
            dcc.Graph(id=name, figure=fig),
        ])

    def run(self):
        self.app.run_server(debug=True)
