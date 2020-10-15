 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
from layout_2 import generate_table, plot_regr,analyse_data
from app import app




@app.callback( Output('data_dropdown', 'children'),
    [Input('show_data', 'value')])

def show_data(value):
    return plot_regr(value)

@app.callback( Output('menu-output-container', 'children'),
    [Input('exo-choice', 'value')])

def show_data(value):
    return analyse_data(value)


@app.callback(
    Output('page', 'pathname'),
    [Input('exo-choice', 'value')
     ])

def change_url(year):
    return '/apps/' + str(year)
