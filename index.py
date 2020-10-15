#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from app import app
from layout_2 import menu

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
])
server = app.server


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    return menu()


if __name__ == '__main__':
    app.run_server(debug=True)
