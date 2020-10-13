#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
from layout_2 import generate_table, choose_year, update_layout
from app import app


@app.callback(
    Output('dd-output-container', 'children'),
    [Input('year-dropdown', 'value'),
     Input('abs-choice','value')])

def display_value(year,indicator):
    '''changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'year-dropdown'in changed_id:
        return update_layout(year)
    elif 'abs-choice'in changed_id:'''
    return update_layout(year,indicator)


@app.callback(
    Output('page', 'pathname'),
    [Input('year-dropdown', 'value')
     ])

def change_url(year):
    return '/apps/' + str(year)
